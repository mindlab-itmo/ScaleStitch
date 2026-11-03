"""Prepare, run and score the vanilla ordinary-label text baseline."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from . import text_baseline_runtime as runtime
from .runtime_provenance import RuntimeProvenance


def load_tokenizers(model_path):
    from transformers import AutoTokenizer

    base = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
    comparison = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
    if not base.chat_template:
        raise ValueError("the local base tokenizer needs its original chat template")
    comparison.chat_template = base.chat_template
    comparison.pad_token = comparison.eos_token
    tokens = ["<OBJ%03d>" % index for index in range(150)]
    if comparison.add_tokens(tokens, special_tokens=True) != 150:
        raise ValueError("base tokenizer already contains native object tokens")
    native = {comparison.convert_tokens_to_ids(token) for token in tokens}
    if len(native) != 150 or min(native) != len(base):
        raise ValueError("ordinary and native object vocabularies overlap")
    configuration = json.loads(Path(model_path, "generation_config.json").read_text(encoding="utf-8"))
    eos = configuration["eos_token_id"]
    eos = [eos] if isinstance(eos, int) else eos
    runtime.generation_settings(eos)
    stamp = dict(base_vocab=len(base), adapted_vocab=len(comparison),
        base_chat_template_sha256=hashlib.sha256(base.chat_template.encode("utf-8")).hexdigest(),
        native_object_token_ids=sorted(native), common_eos_token_ids=eos,
        base_tokenizer_sha256=runtime.digest(base.get_vocab()), adapted_tokenizer_sha256=runtime.digest(comparison.get_vocab()))
    return base, comparison, native, stamp


def source_identity(config, provenance, tokenizer_stamp):
    effective = dict(config, host={"model": {"llama_model_path": config["model"]["local_path"]}})
    provenance.configure(effective)
    provenance.input("cohort", config["cohort"])
    if "measurement_source" in config:
        provenance.input("measurement_source", config["measurement_source"])
    provenance.save()
    return dict(protocol=runtime.PROTOCOL, arm="base", configuration=config,
        effective_config_sha256=provenance.state["effective_config_sha256"],
        sources_sha256=provenance.state["inputs"], code_sha256=provenance.state["code_sha256"],
        environment=provenance.state["environment"], tokenizers=tokenizer_stamp,
        conditions=list(runtime.CONDITIONS), generation=runtime.generation_settings(tokenizer_stamp["common_eos_token_ids"]))


@contextmanager
def output_lock(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "LOCK").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            acquire = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            acquire = lambda: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(stream, fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as error:
            raise RuntimeError("another baseline invocation owns this output") from error
        try:
            yield
        finally:
            stream.seek(0)
            release()


def load_model(config):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(config["model"]["local_path"],
        local_files_only=True, torch_dtype=torch.float16, attn_implementation="sdpa")
    model.to(config["device"]).requires_grad_(False).eval()
    dtypes = sorted({str(value.dtype) for value in model.parameters() if value.is_floating_point()})
    if dtypes != ["torch.float16"]:
        raise ValueError("the vanilla baseline requires actual FP16 model parameters")
    audit = dict(arm="base", parameter_dtypes=dtypes, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        attention=model.config._attn_implementation, output_vocab=model.get_output_embeddings().weight.shape[0],
        device=str(next(model.parameters()).device), adapted=False)
    return model, audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("action", nargs="?", choices=("describe", "prepare", "generate", "score"), default="describe")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--references", type=Path, help="source-object answers; read only by score")
    args = parser.parse_args(argv)
    if args.action == "describe":
        print(json.dumps(dict(protocol=runtime.PROTOCOL, arm="base", status="not_launched")))
        return 0
    if args.config is None or args.output is None:
        parser.error("--config and --output are required")
    if (args.action == "score") != (args.references is not None):
        parser.error("--references is required only for score")
    config = runtime.load_config(args.config)
    import torch
    from .loading import seed_everything

    with output_lock(args.output):
        invocation = args.output / "invocations" / str(time.time_ns())
        invocation.mkdir(parents=True)
        with RuntimeProvenance(invocation, "text-baseline-" + args.action, args.config,
                command=[sys.executable, *sys.argv], torch_module=torch) as provenance:
            tokenizers = load_tokenizers(config["model"]["local_path"])
            identity = source_identity(config, provenance, tokenizers[3])
            runtime.immutable_json(args.output / "manifest.json", identity)
            seed_everything(config["seed"])
            provenance.phase(args.action)
            if args.action == "prepare":
                result = runtime.prepare_inputs(config, args.output, identity, tokenizers)
            elif args.action == "generate":
                runtime.check_prepared(args.output, identity)
                model = None
                if not (args.output / "SUCCESS.json").exists():
                    model, audit = load_model(config)
                    runtime.immutable_json(invocation / "model_audit.json", audit)
                    provenance.record_gpu()
                result = runtime.generate(config, args.output, identity, tokenizers[0], model,
                    tokenizers[3]["common_eos_token_ids"])
            else:
                result = runtime.score(args.output, identity, args.references)
            provenance.result = dict(completed_predictions=result.get("outputs", result.get("inputs")))
            print(json.dumps({key: value for key, value in result.items() if key in
                {"status", "inputs", "outputs", "errors", "questions", "identity_sha256"}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
