"""Prepare identical ordinary-label text for base and adapted Llama models.

This path supplies only chat token IDs: it creates no multimodal object slots,
zeroed features, object descriptions or learned scene prefixes. The adapted
versus base comparison changes the full learned adapter/embedding/head pipeline;
its difference cannot be attributed specifically to arithmetic or to LoRA.
"""
import hashlib
import json
from numbers import Integral
import re

from .packet_scope import FocusedScenePacket
from .scene_extents import SceneHeightPacket
from .scene_measurements import SceneRadiusPacket


_NATIVE_ID = re.compile(r"<OBJ([0-9]{3})>")
_LABEL = re.compile(r"[A-Z]+")
_NATIVE_INSTRUCTION = "Answer with exactly one object identifier and no other text."
_LABEL_INSTRUCTION = "Answer with exactly one label and no other text."


def ordinary_label(index):
    """Map a zero-based visible position to A..Z, AA..AZ, ... (149 -> ET)."""
    if isinstance(index, bool) or not isinstance(index, Integral) or index < 0:
        raise ValueError("label index must be a nonnegative integer")
    value, label = int(index) + 1, ""
    while value:
        value, remainder = divmod(value - 1, 26)
        label = chr(65 + remainder) + label
    return label


def _nonnegative_integer(value):
    return not isinstance(value, bool) and isinstance(value, Integral) and value >= 0


def prepare_example(*, qid, scene_id, question_text, objects, candidate_source_rows, packet=None):
    """Relabel an already visible-ID question and its existing scene packet.

    ``objects`` explicitly binds active visible positions to original source
    rows. ``candidate_source_rows`` uses source rows, whereas ``question_text``
    uses visible IDs after the host's source-to-visible question conversion.
    Packets must already contain their full-scene T/C/S intervention. A focused
    wrapper is accepted without rebuilding its donors or numerical blocks.
    """
    bindings = {}
    sources = set()
    for item in objects:
        if not isinstance(item, dict) or not {"visible_id", "source_row"} <= item.keys():
            raise ValueError("objects require explicit visible_id/source_row bindings")
        visible, source = item["visible_id"], item["source_row"]
        if (not _nonnegative_integer(visible) or visible > 999 or not _nonnegative_integer(source)
                or visible in bindings or source in sources):
            raise ValueError("visible/source object bindings must be unique nonnegative integers")
        bindings[int(visible)] = int(source)
        sources.add(int(source))
    candidates = tuple(candidate_source_rows)
    if (not bindings or len(candidates) != 2 or any(not _nonnegative_integer(v) for v in candidates)
            or len(set(candidates)) != 2 or not set(candidates) <= sources):
        raise ValueError("exactly two distinct visible candidate source rows are required")
    if not isinstance(question_text, str) or not question_text.strip():
        raise ValueError("a nonempty visible-ID question is required")
    matches = list(_NATIVE_ID.finditer(question_text))
    query_ids = [int(match.group(1)) for match in matches]
    if (question_text.count("<OBJ") != len(matches) or len(query_ids) != 2
            or len(set(query_ids)) != 2 or not set(query_ids) <= bindings.keys()
            or {bindings[v] for v in query_ids} != set(candidates)):
        raise ValueError("question must name exactly the two visible candidate IDs")
    if (question_text.count(_NATIVE_INSTRUCTION) != 1
            or not question_text.endswith(_NATIVE_INSTRUCTION)):
        raise ValueError("the original canonical object-identifier answer instruction is required")

    def relabel(text):
        def replacement(match):
            visible = int(match.group(1))
            if visible not in bindings:
                raise ValueError("packet contains an unknown visible object ID")
            return "Object " + ordinary_label(visible)
        result = _NATIVE_ID.sub(replacement, text)
        if "<OBJ" in result:
            raise ValueError("malformed native object identifier")
        return result

    question = relabel(question_text).replace(_NATIVE_INSTRUCTION, _LABEL_INSTRUCTION)
    packet_text, packet_metadata, packet_hash = "", None, None
    if packet is not None:
        if not isinstance(packet, (SceneRadiusPacket, SceneHeightPacket, FocusedScenePacket)):
            raise TypeError("an already constructed scene measurement packet is required")
        records = tuple(packet.records)
        record_ids = [record.visible_id for record in records]
        if (not records or len(set(record_ids)) != len(records)
                or any(bindings.get(record.visible_id) != record.source_row for record in records)
                or not set(candidates) <= {record.source_row for record in records}
                or [int(match.group(1)) for match in _NATIVE_ID.finditer(packet.text)] != record_ids):
            raise ValueError("packet records disagree with visible/source object bindings")
        # Only ID notation changes. In particular numerical/unit byte strings
        # and full-scene reassignment donors are never reconstructed.
        packet_text = relabel(packet.text)
        packet_metadata = packet.metadata()
        packet_hash = hashlib.sha256(packet.text.encode("utf-8")).hexdigest()
    label_to_source = {ordinary_label(v): bindings[v] for v in sorted(bindings)}
    source_to_label = {source: label for label, source in label_to_source.items()}
    return dict(qid=qid, scene_id=scene_id, user_text=question + packet_text,
                question_text=question, packet_text=packet_text, packet_metadata=packet_metadata,
                source_packet_sha256=packet_hash, label_to_source_row=label_to_source,
                candidate_labels=[source_to_label[source] for source in candidates],
                input_mode="text_only", packet_placement="question_tail",
                comparison_scope="full learned adapter/embedding/head pipeline; no arithmetic-only or LoRA-only attribution")


def shared_prefix(base_tokenizer, adapted_tokenizer, user_text):
    """Block comparison unless independently rendered UTF-8 text and IDs agree."""
    if not isinstance(user_text, str) or not user_text.strip() or "<OBJ" in user_text:
        raise ValueError("text-only input must use ordinary labels without native object IDs")
    rendered, encoded = [], []
    for tokenizer in (base_tokenizer, adapted_tokenizer):
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("an explicit shared base chat template is required")
        text = tokenizer.apply_chat_template([dict(role="user", content=user_text)],
            tokenize=False, add_generation_prompt=True)
        if not isinstance(text, str) or "<OBJ" in text:
            raise ValueError("chat rendering must return ordinary-label text")
        rendered.append(text)
    if rendered[0].encode("utf-8") != rendered[1].encode("utf-8"):
        raise ValueError("base/adapted rendered chat text bytes disagree")
    for tokenizer, text in zip((base_tokenizer, adapted_tokenizer), rendered):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if (not isinstance(ids, (list, tuple)) or not ids
                or any(not _nonnegative_integer(value) for value in ids)):
            raise ValueError("tokenization must return one flat nonempty token ID sequence")
        encoded.append([int(value) for value in ids])
    if encoded[0] != encoded[1]:
        raise ValueError("base/adapted chat token IDs disagree")
    return dict(rendered_text=rendered[0], input_ids=encoded[0],
                rendered_text_sha256=hashlib.sha256(rendered[0].encode("utf-8")).hexdigest(),
                input_ids_sha256=hashlib.sha256(json.dumps(encoded[0], separators=(",", ":")).encode()).hexdigest(),
                prefix_tokens=len(encoded[0]))


def score_output(text, *, label_to_source_row, candidate_labels, target_source_row):
    """Score a whole uppercase label, optionally followed by one full stop."""
    labels = dict(label_to_source_row)
    candidates = tuple(candidate_labels)
    if (not labels or any(not isinstance(label, str) or _LABEL.fullmatch(label) is None
                          or not _nonnegative_integer(source) for label, source in labels.items())
            or len(set(labels.values())) != len(labels) or len(candidates) != 2
            or len(set(candidates)) != 2 or not set(candidates) <= labels.keys()
            or not _nonnegative_integer(target_source_row)
            or target_source_row not in {labels[label] for label in candidates}):
        raise ValueError("scoring requires a bijective label map, two candidates and their target source row")
    if not isinstance(text, str):
        raise ValueError("the raw model output must be text")
    match = re.fullmatch(r"([A-Z]+)\.?", text.strip())
    label = None if match is None else match.group(1)
    visible = label in labels
    compliant = visible and label in candidates
    source = labels[label] if visible else None
    return dict(raw_output=text, label=label, source_row=source, format_valid=match is not None,
                visible_label=visible, candidate_compliant=compliant,
                correct=compliant and source == target_source_row)
