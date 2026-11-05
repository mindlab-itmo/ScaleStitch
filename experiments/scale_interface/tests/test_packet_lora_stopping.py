"""Scripted-token regression for the observed partial host-ending failure."""
import json
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest

import torch
from transformers import StoppingCriteriaList

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scale_interface import natural_qa as qa
from scale_interface import packet_lora as lora
from scale_interface import packet_lora_cli as cli


class EndingTokenizer:
    """The relevant actual Llama-3 token boundaries; no model or data files."""
    all_special_ids = (128256, 128257, 128258)

    def __call__(self, text, **kwargs):
        parts = re.findall(r"<OBJ[0-9]{3}>|</|.", text, flags=re.S)
        return {"input_ids": [128256 + int(part[4:7]) if part.startswith("<OBJ")
                              else {"</": 524, "s": 82, ">": 29}.get(part, ord(part))
                              if len(part) == 1 else 524 for part in parts]}

    def decode(self, ids, skip_special_tokens=False):
        return "".join("" if skip_special_tokens and int(value) in self.all_special_ids
                       else f"<OBJ{int(value) - 128256:03d}>" if int(value) >= 128256
                       else {524: "</", 82: "s", 29: ">"}.get(int(value), chr(int(value))) for value in ids)


class PartialEndingStop:
    def __init__(self):
        self.stops = [torch.tensor(ids) for ids in ([14711], [198, 14711], [82, 29], [524])]

    def __call__(self, ids, scores, **kwargs):
        return any(ids.shape[1] >= len(stop) and torch.all(ids[0, -len(stop):] == stop).item()
                   for stop in self.stops)


class ScriptedGenerator:
    """Consume stopping criteria exactly where generate checks a new token."""
    def __init__(self):
        self.tokens = [128256, 524, 82, 29, 42]

    def generate(self, *, stopping_criteria, **kwargs):
        for length in range(1, len(self.tokens) + 1):
            ids = torch.tensor([self.tokens[:length]])
            if stopping_criteria(ids, None).all():
                return ids
        return ids


def stopping_host():
    return types.SimpleNamespace(end_sym="</s>", llama_tokenizer=EndingTokenizer(),
                                 llama_model=ScriptedGenerator())


class PacketLoraStoppingTests(unittest.TestCase):
    def test_other_legacy_and_custom_stops_are_retained_without_mutation(self):
        core = stopping_host()
        legacy = PartialEndingStop()
        extra = lambda ids, scores, **kwargs: ids[:, -1] == 77
        original_stops = [stop.clone() for stop in legacy.stops]
        with lora.complete_answer_stopping(core):
            for script, expected in (([128256, 14711, 42], [[128256, 14711]]),
                                     ([128256, 77, 42], [[128256, 77]])):
                core.llama_model.tokens = script
                self.assertEqual(core.llama_model.generate(stopping_criteria=StoppingCriteriaList([legacy, extra])).tolist(), expected)
        self.assertTrue(all(torch.equal(old, current) for old, current in zip(original_stops, legacy.stops)))

    def test_ending_override_restores_legacy_generation_even_after_an_exception(self):
        core = stopping_host()
        legacy = StoppingCriteriaList([PartialEndingStop()])
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with lora.complete_answer_stopping(core):
                self.assertEqual(core.llama_model.generate(stopping_criteria=legacy).tolist(), [[128256, 524, 82, 29]])
                raise RuntimeError("interrupted")
        self.assertEqual(core.llama_model.generate(stopping_criteria=legacy).tolist(), [[128256, 524]])

    def test_evaluation_finishes_the_learned_ending_and_keeps_native_object_tokens(self):
        core = stopping_host()
        cpu = {"assigned_ids": torch.tensor([[2, 0, 1]]), "scene_mask": torch.ones(1, 3, dtype=torch.bool)}
        def inputs(example, representation):
            condition = qa.NaturalQACondition(representation=representation)
            return "q0", "synthetic_00", "question", cpu, cpu, condition, None, 2, "radius"
        def run_one(core, batch, *args, **kwargs):
            tokens = core.llama_model.generate(stopping_criteria=StoppingCriteriaList([PartialEndingStop()]))
            return None, qa._decoded_prediction(core, tokens[0], batch)
        identity = dict(arm="lexical", source_hashes={"source": "a" * 64},
                        input_hashes={"inputs": "b" * 64}, protocol_hash="c" * 64)
        proof = dict(path="checkpoint.pt", sha256="d" * 64, row_position=3504, step=219)
        with tempfile.TemporaryDirectory() as folder:
            result = cli.evaluate(core, [("radius", 0, 2)], inputs, Path(folder), identity, proof, run_one=run_one)
            self.assertEqual(result["counts"], {"absent": {"n": 1, "correct": 1}, "lexical": {"n": 1, "correct": 1}})
            manifest = json.loads((Path(folder) / "manifest.json").read_text())
            self.assertEqual(manifest["answer_stopping"], dict(protocol="host_stops_without_partial_ending/v2",
                end_sym="</s>", ending_token_ids=[524, 82, 29], removed_stop_token_sequences=[[524]],
                retained_legacy_stop_token_sequences=[[14711], [198, 14711], [82, 29]]))


if __name__ == "__main__":
    unittest.main()
