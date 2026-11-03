"""Vanilla text baseline with fixed packet controls and resumable records."""
from collections import Counter, defaultdict
from contextlib import nullcontext
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import hashlib

from .runtime_provenance import sha256

CONDITIONS = ("absent", "full_true", "full_constant", "full_swapped",
              "pair_true", "pair_constant", "pair_swapped")
TASKS = ("radius", "height")
PROTOCOL = "ordinary-label-text-only/v1"

def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + str(os.getpid()) + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != encoded(value):
            raise ValueError("immutable manifest differs: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + str(os.getpid()) + "." + str(time.time_ns()) + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded(value))
            stream.flush()
            os.fsync(stream.fileno())
        # Same-directory hardlink publishes all bytes atomically and refuses
        # replacement. Unlike direct exclusive creation, a crash cannot expose
        # a half-written immutable identity or success receipt.
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != encoded(value):
                raise ValueError("immutable manifest differs: " + str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


class DurableStore:
    """One transaction per immutable record; interruption never drops a cohort row."""
    def __init__(self, path, identity, *, read_only=False):
        self.read_only = read_only
        if read_only:
            self.connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=60)
            self.connection.execute("PRAGMA query_only=ON")
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(str(path), timeout=60)
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
            self.connection.execute("CREATE TABLE IF NOT EXISTS identity (value TEXT NOT NULL)")
            self.connection.execute("CREATE TABLE IF NOT EXISTS records (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self.connection.execute("CREATE TABLE IF NOT EXISTS failures (id INTEGER PRIMARY KEY, key TEXT, value TEXT)")
        wanted = encoded(identity).decode("utf-8")
        rows = self.connection.execute("SELECT value FROM identity").fetchall()
        if not rows and not read_only:
            self.connection.execute("INSERT INTO identity VALUES (?)", (wanted,))
            self.connection.commit()
        elif rows != [(wanted,)]:
            self.connection.close()
            raise ValueError("durable store identity differs")

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.connection.close()

    def contains(self, key):
        return self.connection.execute("SELECT 1 FROM records WHERE key=?", (key,)).fetchone() is not None

    def add(self, key, value):
        if self.read_only:
            raise ValueError("read-only prepared/collected store cannot be mutated")
        text = encoded(value).decode("utf-8")
        previous = self.connection.execute("SELECT value FROM records WHERE key=?", (key,)).fetchone()
        if previous is not None:
            if previous[0] != text:
                raise ValueError("immutable duplicate record differs: " + key)
            return
        with self.connection:
            self.connection.execute("INSERT INTO records VALUES (?,?)", (key, text))

    def rows(self):
        for (value,) in self.connection.execute("SELECT value FROM records ORDER BY key"):
            yield json.loads(value)

    def count(self):
        return self.connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]

    def get(self, key):
        row = self.connection.execute("SELECT value FROM records WHERE key=?", (key,)).fetchone()
        if row is None:
            raise ValueError("required completed record is absent: " + key)
        return json.loads(row[0])

    def add_failure(self, key, value):
        if self.read_only:
            raise ValueError("read-only store cannot record a failure")
        with self.connection:
            self.connection.execute("INSERT INTO failures (key,value) VALUES (?,?)",
                (key, encoded(dict(value, recorded_utc=now())).decode("utf-8")))

    def failure_count(self):
        return self.connection.execute("SELECT COUNT(*) FROM failures").fetchone()[0]


def candidate_rows(original_prompt):
    if not isinstance(original_prompt, str):
        raise ValueError("original prompt required")
    rows = tuple(int(x) for x in re.findall(r"<OBJ([0-9]{3})>", original_prompt))
    if len(rows) != 2 or len(set(rows)) != 2 or original_prompt.count("<OBJ") != 2:
        raise ValueError("original prompt must name exactly two distinct source rows")
    return rows


def prepare_pair(*, qid, scene_id, task, original_prompt, visible_question, batch,
                 base_tokenizer, adapted_tokenizer, native_object_token_ids,
                 measurement_source=None):
    from experiments.scale_interface.natural_qa import NaturalQACondition, _packet
    from experiments.scale_interface.packet_scope import focus_scene_packet
    from experiments.scale_interface.text_only_baseline import prepare_example, shared_prefix
    if task not in TASKS:
        raise ValueError("only direct radius and height are in this protocol")
    candidates = candidate_rows(original_prompt)
    assignment = batch["assigned_ids"][0].tolist()
    mask = batch["scene_mask"][0].tolist()
    if len(assignment) != len(mask):
        raise ValueError("assignment/mask size differs")
    objects = [dict(visible_id=v, source_row=int(assignment[v]))
               for v, active in enumerate(mask) if active]
    # Construct each scene-wide intervention first. Pair scope is only a view
    # over its records; never construct swapped values from the two candidates.
    full = {mode: _packet(batch, NaturalQACondition(measurement=task, mode=mode), 100,
        measurement_source=measurement_source, scene_id=scene_id)
        for mode in ("true", "constant", "swapped")}
    packets = [("absent", None)]
    packets += [("full_" + mode, full[mode]) for mode in full]
    packets += [("pair_" + mode, focus_scene_packet(full[mode], candidate_source_rows=candidates))
                for mode in full]
    result = []
    for condition, packet in packets:
        item = prepare_example(qid=qid, scene_id=scene_id, question_text=visible_question,
            objects=objects, candidate_source_rows=candidates, packet=packet)
        item.update(shared_prefix(base_tokenizer, adapted_tokenizer, item["user_text"]))
        if set(item["input_ids"]) & set(native_object_token_ids):
            raise ValueError("native object token IDs reached ordinary text input")
        item.update(task=task, condition=condition, candidate_source_rows=list(candidates), error=None)
        item["input_sha256"] = digest(item)
        result.append(item)
    return result


def generation_settings(eos_token_ids):
    eos = list(eos_token_ids)
    if (not eos or len(set(eos)) != len(eos)
            or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in eos)):
        raise ValueError("distinct ordinary integer base EOS IDs are required")
    return dict(do_sample=False, num_beams=1, max_new_tokens=32, min_new_tokens=0,
        repetition_penalty=1., use_cache=True, eos_token_id=eos, pad_token_id=eos[0],
        bos_token_id=None, forced_bos_token_id=None, forced_eos_token_id=None,
        no_repeat_ngram_size=0, renormalize_logits=False)


def finite_logits(input_ids, scores):
    import torch
    if not torch.isfinite(scores).all():
        raise ValueError("nonfinite language-model logits; scientific run is incomplete")
    return scores


def decode_new_tokens(tokenizer, full_ids, prefix_ids, eos_token_ids):
    prefix = list(prefix_ids)
    sequence = list(full_ids)
    if sequence[:len(prefix)] != prefix:
        raise ValueError("generated token prefix differs from the exact shared input")
    generated = sequence[len(prefix):]
    score_ids = list(generated)
    # Never skip all special tokens: a generated <OBJ...> remains invalid text.
    # Only one terminal common base EOS is an end marker rather than an answer.
    if score_ids and score_ids[-1] in eos_token_ids:
        score_ids.pop()
    return dict(output_token_ids=generated, decoded_token_ids=score_ids,
        raw_output=tokenizer.decode(score_ids, skip_special_tokens=False,
            clean_up_tokenization_spaces=False),
        raw_output_with_eos=tokenizer.decode(generated, skip_special_tokens=False,
            clean_up_tokenization_spaces=False))


def score_record(record, target_source_row):
    from experiments.scale_interface.text_only_baseline import score_output
    failed = bool(record.get("error"))
    text = record.get("raw_output", "") if not failed else ""
    options = dict(label_to_source_row=record["label_to_source_row"],
        candidate_labels=record["candidate_labels"], target_source_row=target_source_row)
    primary = score_output(text, **options)
    match = re.fullmatch(r"(?:Object )?([A-Z]+)", text.strip(), flags=re.IGNORECASE)
    secondary = score_output(match.group(1).upper() if match else "", **options)
    return dict(primary=primary, whole_label_sensitivity=secondary, error=failed)


def record_key(task, qid, condition):
    return json.dumps([task, str(qid), condition], separators=(",", ":"))


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "task", "model", "data", "cohort", "seed", "device"}
    if not isinstance(config, dict) or set(config) - (required | {"measurement_source"}) or not required <= config.keys():
        raise ValueError("unknown or missing text-baseline configuration fields")
    if (config["schema_version"] != "scalestitch_text_baseline/v1" or config["task"] not in TASKS
            or type(config["seed"]) is not int or config["seed"] < 0
            or re.fullmatch(r"cpu|cuda(?::[0-9]+)?", str(config["device"])) is None
            or not isinstance(config["model"], dict) or set(config["model"]) != {"local_path"}):
        raise ValueError("invalid fixed text-baseline configuration")

    def resolve(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("explicit local input paths are required")
        return str((path.parent / value).resolve())

    data = config["data"]
    paths = {"features", "radii", "attributes", "rows", "images", "relations", "full_manifest"}
    if (not isinstance(data, dict) or set(data) - (paths | {"feature_dim", "point_cloud_type"})
            or not {"features", "radii", "attributes", "rows"} <= data.keys()):
        raise ValueError("data must use the SceneDataset configuration")
    config["data"] = {key: resolve(value) if key in paths and value is not None else value
                      for key, value in data.items()}
    config["model"] = {"local_path": resolve(config["model"]["local_path"])}
    config["cohort"] = resolve(config["cohort"])
    if (config["task"] == "height") != ("measurement_source" in config):
        raise ValueError("height alone requires an explicit measurement_source")
    if "measurement_source" in config:
        config["measurement_source"] = resolve(config["measurement_source"])
    return config


def source_rows(config):
    cohort = json.loads(Path(config["cohort"]).read_text(encoding="utf-8"))
    raw = json.loads(Path(config["data"]["rows"]).read_text(encoding="utf-8"))
    valid_id = lambda x: type(x) in (str, int) and bool(str(x).strip())
    if (not isinstance(cohort, list) or not cohort or not all(valid_id(q) for q in cohort)
            or len(set(map(str, cohort))) != len(cohort) or not isinstance(raw, list) or not raw):
        raise ValueError("a nonempty saved cohort with unique question IDs is required")
    indexed = {}
    for row in raw:
        if not isinstance(row, dict) or not valid_id(row.get("qid")) or str(row["qid"]) in indexed:
            raise ValueError("source question IDs must be unique")
        indexed[str(row["qid"])] = row
    if not set(map(str, cohort)) <= indexed.keys():
        raise ValueError("saved cohort contains a missing source question")
    return [{key: indexed[str(qid)][key] for key in ("qid", "scene_id", "prompt", "assigned_ids")}
            for qid in cohort]


def _verify_matrix(rows, cohort):
    expected = {(str(qid), str(scene), condition) for qid, scene in cohort for condition in CONDITIONS}
    actual = [(str(row["qid"]), str(row["scene_id"]), row["condition"]) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected or any(row.get("error") for row in rows):
        raise ValueError("complete fixed-cohort seven-condition matrix is required")
    for row in rows:
        if row["input_sha256"] != digest({key: value for key, value in row.items() if key != "input_sha256"}):
            raise ValueError("prepared input record differs")


def check_prepared(output, identity):
    output = Path(output)
    if json.loads((output / "manifest.json").read_text(encoding="utf-8")) != identity:
        raise ValueError("prepared run identity differs")
    receipt = json.loads((output / "PREPARED.json").read_text(encoding="utf-8"))
    if receipt["status"] != "complete" or receipt["identity_sha256"] != digest(identity) or receipt["errors"] != 0:
        raise ValueError("prepared input receipt is incomplete or changed")
    with DurableStore(output / "inputs.sqlite3", identity, read_only=True) as store:
        rows = list(store.rows())
    _verify_matrix(rows, receipt["cohort"])
    if len(rows) != receipt["inputs"] or digest([digest(row) for row in rows]) != receipt["input_collection_sha256"]:
        raise ValueError("prepared input collection differs")
    return receipt, rows


def prepare_inputs(config, output, identity, tokenizers):
    from .data import SceneDataset
    from .natural_qa import _cpu_input
    from .scene_extents import normalize_measurement_source

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    immutable_json(output / "manifest.json", identity)
    base, comparison, native, stamp = tokenizers
    immutable_json(output / "tokenizers.json", stamp)
    if (output / "PREPARED.json").exists():
        return check_prepared(output, identity)[0]
    rows = source_rows(config)
    dataset = SceneDataset(**dict(config["data"], rows=rows))
    declaration = None
    if config["task"] == "height":
        declaration = normalize_measurement_source(json.loads(Path(config["measurement_source"]).read_text(encoding="utf-8")))
        if (declaration["input_id"] != sha256(config["data"]["attributes"])
                or any(declaration["verified_scenes"].get(row["scene_id"]) is not True for row in rows)):
            raise ValueError("height declaration does not match the complete source cohort")
    cohort = [[str(row["qid"]), str(row["scene_id"])] for row in rows]
    with DurableStore(output / "inputs.sqlite3", identity) as store:
        for index, row in enumerate(rows):
            keys = [record_key(config["task"], row["qid"], c) for c in CONDITIONS]
            if all(store.contains(key) for key in keys):
                continue
            try:
                qid, scene, question, batch = _cpu_input(dataset, index)
                records = prepare_pair(qid=qid, scene_id=scene, task=config["task"], original_prompt=row["prompt"],
                    visible_question=question, batch=batch, base_tokenizer=base, adapted_tokenizer=comparison,
                    native_object_token_ids=native, measurement_source=declaration)
            except Exception as error:
                store.add_failure(keys[0], {"error": type(error).__name__ + ": " + str(error)})
                raise RuntimeError("technical preparation failure; fixed cohort remains incomplete") from error
            for key, record in zip(keys, records):
                store.add(key, record)
        records = list(store.rows())
        _verify_matrix(records, cohort)
    receipt = dict(status="complete", identity_sha256=digest(identity), inputs=len(rows) * 7, errors=0,
        cohort=cohort, input_collection_sha256=digest([digest(row) for row in records]), tokenizers_sha256=digest(stamp))
    immutable_json(output / "PREPARED.json", receipt)
    return receipt


def validate_outputs(records, inputs):
    wanted = {record_key(row["task"], row["qid"], row["condition"]): row for row in inputs}
    seen = set()
    for row in records:
        key = record_key(row["task"], row["qid"], row["condition"])
        if key not in wanted or key in seen or row["arm"] != "base" or row.get("error"):
            raise ValueError("complete base-only output matrix is required")
        if (not isinstance(row["raw_output"], str) or any(row[name] != wanted[key][name] for name in
                ("scene_id", "input_sha256", "rendered_text_sha256", "input_ids_sha256"))):
            raise ValueError("output differs from its prepared input binding")
        seen.add(key)
    if seen != wanted.keys():
        raise ValueError("output matrix is incomplete")


def _completed_outputs(output, identity, prepared, inputs):
    output = Path(output)
    invocation = json.loads((output / "invocation.json").read_text(encoding="utf-8"))
    success = json.loads((output / "SUCCESS.json").read_text(encoding="utf-8"))
    if (invocation["identity_sha256"] != digest(identity)
            or invocation["prepared_sha256"] != sha256(output / "PREPARED.json")
            or success["identity_sha256"] != digest(identity) or success["errors"] != 0):
        raise ValueError("completed output identity differs")
    with DurableStore(output / "outputs.sqlite3", invocation, read_only=True) as store:
        records = list(store.rows())
    validate_outputs(records, inputs)
    if (len(records) != prepared["inputs"] or len(records) != success["outputs"]
            or any(row.get("error") for row in records) or digest([digest(row) for row in records]) != success["outputs_sha256"]):
        raise ValueError("all fixed-cohort successful forwards are required, including invalid answers")
    return success, records


def generate(config, output, identity, tokenizer, model, eos):
    import torch
    from transformers import GenerationConfig, LogitsProcessorList
    from .text_only_baseline import shared_prefix

    output = Path(output)
    prepared, inputs = check_prepared(output, identity)
    stamp = json.loads((output / "tokenizers.json").read_text(encoding="utf-8"))
    if stamp["common_eos_token_ids"] != list(eos) or digest(stamp) != prepared["tokenizers_sha256"]:
        raise ValueError("generation EOS or tokenizer stamp differs")
    invocation = dict(identity_sha256=digest(identity), prepared_sha256=sha256(output / "PREPARED.json"),
        generation=generation_settings(eos))
    immutable_json(output / "invocation.json", invocation)
    if (output / "SUCCESS.json").exists():
        return _completed_outputs(output, identity, prepared, inputs)[0]
    cuda = str(config["device"]).startswith("cuda")
    with DurableStore(output / "outputs.sqlite3", invocation) as store:
        for source in inputs:
            key = record_key(source["task"], source["qid"], source["condition"])
            if store.contains(key):
                continue
            record = {key: source[key] for key in ("task", "qid", "scene_id", "condition", "input_sha256",
                "rendered_text_sha256", "input_ids_sha256")}
            record.update(arm="base", error=None, raw_output="")
            started = time.monotonic()
            try:
                prefix = shared_prefix(tokenizer, tokenizer, source["user_text"])
                if any(prefix[k] != source[k] for k in prefix):
                    raise ValueError("prepared text or token IDs changed before inference")
                if len(source["input_ids"]) + 32 > model.config.max_position_embeddings:
                    raise ValueError("complete input exceeds model context; truncation is prohibited")
                ids = torch.tensor([source["input_ids"]], dtype=torch.long, device=config["device"])
                with torch.inference_mode(), (torch.autocast("cuda", dtype=torch.float16) if cuda else nullcontext()):
                    generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                        generation_config=GenerationConfig(**generation_settings(eos)),
                        logits_processor=LogitsProcessorList([finite_logits]))
                if cuda:
                    torch.cuda.synchronize(config["device"])
                record.update(decode_new_tokens(tokenizer, generated[0].tolist(), source["input_ids"], eos))
            except Exception as error:
                record["error"] = type(error).__name__ + ": " + str(error)
                store.add_failure(key, record)
                raise RuntimeError("technical inference failure; failed record remains pending for resume") from error
            record["elapsed_seconds"] = time.monotonic() - started
            store.add(key, record)
            atomic_json(output / "progress.json", dict(completed=store.count(), expected=prepared["inputs"], updated_utc=now()))
        records = list(store.rows())
    validate_outputs(records, inputs)
    success = dict(status="complete", identity_sha256=digest(identity), outputs=len(records), errors=0,
        outputs_sha256=digest([digest(row) for row in records]))
    immutable_json(output / "SUCCESS.json", success)
    return _completed_outputs(output, identity, prepared, inputs)[0]


def score(output, identity, references):
    from .text_baseline_statistics import FAMILIES, bootstrap_interactions

    output = Path(output)
    prepared, inputs = check_prepared(output, identity)
    success, records = _completed_outputs(output, identity, prepared, inputs)
    raw = json.loads(Path(references).read_text(encoding="utf-8"))
    indexed = {str(row["question_id"]): row for row in raw}
    if len(indexed) != len(raw):
        raise ValueError("duplicate scoring reference")
    source_map = {record_key(row["task"], row["qid"], row["condition"]): row for row in inputs}
    counts = defaultdict(Counter)
    endpoints = ("primary", "whole_label_sensitivity")
    paired = {endpoint: {} for endpoint in endpoints}
    for row in records:
        reference = indexed[str(row["qid"])]
        if str(reference["scene_id"]) != row["scene_id"]:
            raise ValueError("reference scene differs from the prepared cohort")
        answers = reference["answers"]
        targets = {int(match.group(1)) for answer in answers
                   for match in [re.fullmatch(r"<OBJ([0-9]{3})>", answer.strip())] if match}
        if len(targets) != 1 or len(targets) != len(set(answers)):
            raise ValueError("one exact source-object reference is required")
        source = source_map[record_key(row["task"], row["qid"], row["condition"])]
        scores = score_record(dict(source, raw_output=row["raw_output"], error=row["error"]), targets.pop())
        counter = counts[row["condition"]]
        counter["N"] += 1
        counter["errors"] += bool(row["error"])
        for metric in endpoints:
            for name in ("correct", "format_valid", "visible_label", "candidate_compliant"):
                counter[metric + "_" + name] += scores[metric][name]
            item = paired[metric].setdefault(str(row["qid"]), dict(qid=str(row["qid"]), scene_id=row["scene_id"], scores={}))
            if item["scene_id"] != row["scene_id"]:
                raise ValueError("one paired question spans scenes")
            item["scores"][row["condition"]] = int(scores[metric]["correct"])
    n = len(prepared["cohort"])
    if set(counts) != set(CONDITIONS) or any(counts[c]["N"] != n for c in CONDITIONS):
        raise ValueError("scoring denominator changed")
    report = dict(protocol=PROTOCOL, arm="base", task=inputs[0]["task"], questions=n, outputs=success["outputs"], errors=0,
        identity_sha256=digest(identity), references_sha256=sha256(references),
        outputs_sha256=success["outputs_sha256"],
        primary="strict uppercase whole label, optional final full stop",
        whole_label_sensitivity="case-insensitive bare label or exact Object LABEL; no punctuation or prose",
        conditions={condition: dict(counts[condition]) for condition in CONDITIONS},
        endpoint_relation="Nonnested endpoints; no union or best-parser selection.",
        scores={endpoint: dict(conditions={condition: {
            name: dict(count=counts[condition][endpoint + "_" + name], denominator=n,
                       fraction=counts[condition][endpoint + "_" + name] / n)
            for name in ("correct", "format_valid", "visible_label", "candidate_compliant")} for condition in CONDITIONS},
            families={name: bootstrap_interactions(list(paired[endpoint].values()), spec["weights"],
                replicates=10000, seed=20260911, family_size=spec["family_size"])
                for name, spec in FAMILIES.items()}) for endpoint in endpoints})
    immutable_json(output / "scores.json", report)
    return report


