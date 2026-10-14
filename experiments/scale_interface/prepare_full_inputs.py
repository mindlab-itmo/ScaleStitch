"""Pack existing GT image/raw-pair features; no feature extraction or GPU use."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from .data import SceneDataset
from .scene_data import BaseDataset


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _path(value, parent):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("input paths must be nonempty strings")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def _vector(value, width, label):
    if (not isinstance(value, torch.Tensor) or value.shape != (width,)
            or not value.is_floating_point() or not torch.isfinite(value).all()
            or not torch.any(value != 0)):
        raise ValueError(f"invalid {label}: expected finite nonzero floating [{width}]")
    return value.float()


def _attribute_identity(attr, scene):
    if not isinstance(attr, dict):
        raise ValueError(f"missing GT attributes for {scene}")
    locs = attr.get("locs")
    if (not isinstance(locs, torch.Tensor) or locs.ndim != 2 or locs.shape[1] != 6
            or not locs.is_floating_point() or not torch.isfinite(locs).all()
            or locs.shape[0] < 3):
        raise ValueError(f"GT locations must be finite [N,6], N >= 3: {scene}")
    ids = [BaseDataset._integer_object_id(value)
           for value in attr.get("obj_ids", range(len(locs)))]
    if len(ids) != len(locs) or len(set(ids)) != len(ids):
        raise ValueError(f"GT row IDs must be unique and aligned: {scene}")
    return ids, locs.float()


def _missing_mask(declarations, scene, count):
    """Validate the fixed declaration in source-row order, never inferred ID order."""
    mask = declarations.get(scene)
    if not isinstance(mask, dict) or set(mask) != {"image_rows", "relation_pairs"}:
        raise ValueError(f"missing-channel declaration requires both masks for {scene}")
    images, pairs = mask["image_rows"], mask["relation_pairs"]
    if (not isinstance(images, list)
            or any(type(row) is not int or not 0 <= row < count for row in images)
            or len(set(images)) != len(images)):
        raise ValueError(f"invalid missing image source rows for {scene}")
    if (not isinstance(pairs, list)
            or any(not isinstance(pair, list) or len(pair) != 2
                   or any(type(row) is not int or not 0 <= row < count for row in pair)
                   or pair[0] == pair[1] for pair in pairs)
            or len({tuple(pair) for pair in pairs}) != len(pairs)):
        raise ValueError(f"invalid missing directed source-row pairs for {scene}")
    return set(images), {tuple(pair) for pair in pairs}


def prepare_full_inputs(config_path, output):
    """Resolve relative paths against the caller's working directory; output is new.

    Preparation config requires base_config, source_attributes, image_features,
    raw_relations, allowed_scenes. allowed_scenes points to a fixed JSON list.
    Raw relation keys accept exactly two spellings: scene_id + '_' + source_id
    + '_' + target_id, or the official scene_id + '\\n_' + source_id + '_'
    + target_id. Both aliases for one pair are ambiguous and rejected.
    Missing pairs stay zero for the host's actual FP32-neighbor preflight; no
    reverse edge is substituted. A present malformed pair, or a retained scene
    with no non-self relations, is an input error before any output is written.

    Optional missing_channel_policy="zero_projected_declared" requires a
    missing_channels JSON path with fixed per-scene image_rows/relation_pairs.
    Only exactly declared absent source entries become zero placeholders; all
    other entries remain required. The emitted v2 manifest and host configuration
    opt into zeroing these slots after projection. Default "reject" emits v1.
    """
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    spec = _read(config_path)
    keys = {"base_config", "source_attributes", "image_features", "raw_relations", "allowed_scenes"}
    optional = {"missing_channel_policy", "missing_channels"}
    if not isinstance(spec, dict) or not keys <= set(spec) or set(spec) - keys - optional:
        raise ValueError(f"preparation config requires {sorted(keys)}; optional fields are {sorted(optional)}")
    policy = spec.get("missing_channel_policy", "reject")
    if policy not in {"reject", "zero_projected_declared"}:
        raise ValueError("invalid missing_channel_policy")
    declared = policy == "zero_projected_declared"
    if declared != ("missing_channels" in spec):
        raise ValueError("zero_projected_declared and missing_channels must be supplied together")
    working_directory = Path.cwd()
    paths = {key: _path(spec[key], working_directory) for key in keys}
    declarations = None
    if declared:
        paths["missing_channels"] = _path(spec["missing_channels"], working_directory)
        declarations = _read(paths["missing_channels"])
        if not isinstance(declarations, dict):
            raise ValueError("missing_channels must map scenes to fixed missing masks")
    config = copy.deepcopy(_read(paths["base_config"]))
    if any(key in config for key in ("codec", "random_codec_seed")):
        raise ValueError("codec settings are not supported by the frozen-host preparation")
    data = config["data"]
    if data.get("point_cloud_type", "gt") != "gt":
        raise ValueError("full preparation requires GT attribute rows")
    for key in ("features", "radii", "attributes", "rows"):
        data[key] = str(_path(data[key], working_directory))
    # Keep an emitted configuration usable after moving out of its source folder.
    for key in ("host_checkpoint", "deployment"):
        if key in config["host"]:
            config["host"][key] = str(_path(config["host"][key], working_directory))
    model = config["host"]["model"]
    model["llama_model_path"] = str(_path(model["llama_model_path"], working_directory))
    if config["host"].get("aqt"):
        raise ValueError("full GT input does not support host.aqt")
    run = config.get("run", {})
    if not isinstance(run.get("conditions"), list) or not run["conditions"]:
        raise ValueError("base_config requires generate-scene run.conditions")
    if run.get("measurement_source"):
        run["measurement_source"] = str(_path(run["measurement_source"], working_directory))
    allowed = _read(paths["allowed_scenes"])
    if (not isinstance(allowed, list) or not allowed
            or any(not isinstance(scene, str) or not scene.strip() for scene in allowed)
            or len(set(allowed)) != len(allowed)):
        raise ValueError("allowed_scenes must contain a nonempty unique list of scene IDs")
    rows = _read(data["rows"])
    if (not isinstance(rows, list) or not rows
            or any(not isinstance(row, dict) or not isinstance(row.get("scene_id"), str)
                   or not isinstance(row.get("qid"), (str, int)) or isinstance(row["qid"], bool)
                   or not str(row["qid"]).strip() for row in rows)
            or len({str(row["qid"]) for row in rows}) != len(rows)):
        raise ValueError("questions must contain scene_id and unique nonempty qid values")
    retained = [row for row in rows if row["scene_id"] in set(allowed)]
    if not retained:
        raise ValueError("fixed scene coverage retains no questions")
    attrs = torch.load(data["attributes"], map_location="cpu", weights_only=True)
    source_attrs = torch.load(paths["source_attributes"], map_location="cpu", weights_only=True)
    video = torch.load(paths["image_features"], map_location="cpu", weights_only=True)
    raw = torch.load(paths["raw_relations"], map_location="cpu", weights_only=True)
    if not all(isinstance(value, dict) for value in (attrs, source_attrs, video, raw)):
        raise ValueError("GT attributes and feature banks must be dictionaries")
    images, relations, identities, absent_pairs, missing_masks = {}, {}, {}, {}, {}
    for scene in sorted({row["scene_id"] for row in retained}):
        ids, locs = _attribute_identity(attrs.get(scene), scene)
        original_ids, original_locs = _attribute_identity(source_attrs.get(scene), scene)
        if ids != original_ids or not torch.equal(locs, original_locs):
            raise ValueError(f"source/working GT row or coordinate identity mismatch: {scene}")
        bank = torch.zeros((len(ids), len(ids), 512), dtype=torch.float32)
        absent_pairs[scene] = 0
        missing_images, missing_pairs = _missing_mask(declarations, scene, len(ids)) if declared else (set(), set())
        for i, source_id in enumerate(ids):
            aliases = sorted({f"{scene}_{source_id}", f"{scene}_{source_id:02d}"} & video.keys())
            if len(aliases) > 1:
                raise ValueError(f"missing or ambiguous image feature: {scene}/{source_id}")
            if i in missing_images:
                if aliases:
                    raise ValueError(f"declared missing image is present: {scene}/{source_id}")
                images[f"{scene}_{source_id}"] = torch.zeros(1024, dtype=torch.float32)
            else:
                if not aliases:
                    raise ValueError(f"missing or undeclared image feature: {scene}/{source_id}")
                images[f"{scene}_{source_id}"] = _vector(video[aliases[0]], 1024, f"image {aliases[0]}")
            for j, target_id in enumerate(ids):
                if i == j:
                    continue
                aliases = [key for key in (f"{scene}_{source_id}_{target_id}",
                                          f"{scene}\n_{source_id}_{target_id}") if key in raw]
                if len(aliases) > 1:
                    raise ValueError(f"ambiguous directed relation aliases: {scene}/{source_id}/{target_id}")
                if (i, j) in missing_pairs:
                    if aliases:
                        raise ValueError(f"declared missing directed relation is present: {scene}/{source_id}/{target_id}")
                    absent_pairs[scene] += 1
                    continue
                if not aliases:
                    if declared:
                        raise ValueError(f"undeclared missing directed relation: {scene}/{source_id}/{target_id}")
                    absent_pairs[scene] += 1
                    continue
                key = aliases[0]
                bank[i, j] = _vector(raw[key], 512, f"directed relation {key}")
        if absent_pairs[scene] == len(ids) * (len(ids) - 1):
            raise ValueError(f"no non-self directed relations for retained scene: {scene}")
        relations[scene], identities[scene] = bank, ids
        if declared:
            missing_masks[scene] = {"image_rows": sorted(missing_images),
                                    "relation_pairs": [list(pair) for pair in sorted(missing_pairs)]}
    input_paths = {**paths, **{key: Path(data[key]) for key in ("features", "radii", "attributes", "rows")},
                   "preparation_config": config_path, "preparer_code": Path(__file__).resolve()}
    provenance = {key: {"path": str(path), "sha256": sha256(path)} for key, path in input_paths.items()}
    output.mkdir(parents=True, exist_ok=False)
    torch.save(images, output / "images.pt")
    torch.save(relations, output / "relations.pt")
    _write(output / "questions.json", retained)
    data.update(images=str(output / "images.pt"), relations=str(output / "relations.pt"),
                rows=str(output / "questions.json"), point_cloud_type="gt")
    manifest = {"schema_version": "scalestitch_full_input/v1",
                "sha256": {key: sha256(data[key]) for key in ("features", "radii", "attributes", "images", "relations")},
                "scene_object_ids": identities, "provenance": provenance,
                "missing_directed_pairs": absent_pairs,
                "relation_preflight": "Actual host FP32 neighbor selection remains required."}
    if declared:
        manifest.update(schema_version="scalestitch_full_input/v2",
                        missing_channel_policy=policy, missing_channels=missing_masks)
    _write(output / "manifest.json", manifest)
    data["full_manifest"] = str(output / "manifest.json")
    # Exercise the real loader and saved object permutations, without a model.
    dataset = SceneDataset(**data)
    for index in range(len(dataset)):
        dataset[index]
    _write(output / "exclusions.json", {
        "candidate_questions": len(rows), "retained_questions": len(retained),
        "excluded_scenes": [{"scene_id": scene, "reason": "outside_fixed_allowed_scenes"}
                            for scene in sorted({row["scene_id"] for row in rows} - set(allowed))],
        "excluded_qids": [row["qid"] for row in rows if row["scene_id"] not in set(allowed)],
        "policy": "Fixed input coverage only; no question/answer or QA outcome selection."})
    model.update(host_mode="full", gt_pretrain=True, add_img_token=True, img_input_dim=1024,
                 knn=2, max_knn=5, add_scene_token=False, bbox_embed=False, bidirection=False,
                 object_feature_intervention="full", missing_channel_policy=policy)
    _write(output / "config.json", config)
    return {"config": str(output / "config.json"), "questions": len(retained), "scenes": len(identities)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_full_inputs(args.config, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
