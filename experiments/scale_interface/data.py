"""Object features, metric radii and explicitly ordered question rows."""
import hashlib
import json
from pathlib import Path

import torch

from .scene_data import BaseDataset, recover_caption, update_caption, train_collate_fn


class SceneDataset(BaseDataset):
    def __init__(self, *, features, radii, attributes, rows, feature_dim=1280,
                 point_cloud_type="gt", images=None, relations=None, full_manifest=None):
        super().__init__()
        supplied = (images is not None, relations is not None, full_manifest is not None)
        if any(supplied) and not all(supplied):
            raise ValueError("full inputs require images, relations and full_manifest together")
        self.full_input = all(supplied)
        self.full_manifest = None
        self.missing_channel_policy = "reject"
        if self.full_input:
            if point_cloud_type != "gt":
                raise ValueError("full inputs currently require GT attribute rows")
            self.full_manifest = self._read_full_manifest(full_manifest, {
                "features": features, "radii": radii, "attributes": attributes,
                "images": images, "relations": relations,
            })
            self.missing_channel_policy = self.full_manifest.get("missing_channel_policy", "reject")
        self.anno = json.loads(Path(rows).read_text()) if isinstance(rows, (str, Path)) else rows
        if not isinstance(self.anno, list) or not self.anno:
            raise ValueError("question rows must be nonempty and have unique qid values")
        question_ids = []
        for row in self.anno:
            qid = row.get("qid") if isinstance(row, dict) else None
            if (isinstance(qid, bool) or not isinstance(qid, (str, int))
                    or not str(qid).strip()):
                raise ValueError("each question row must have a nonempty string or integer qid")
            question_ids.append(str(qid))
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("question rows must have unique qid values")
        self.feats = torch.load(features, map_location="cpu", weights_only=True)
        self.attributes = torch.load(attributes, map_location="cpu", weights_only=True)
        radius_data = torch.load(radii, map_location="cpu", weights_only=True)
        self.radius_records = radius_data.get("records", radius_data)
        if radius_data.get("schema_version") == "scalestitch_object_radius/v3":
            self.radius_epsilon = 1e-6
        self.feat_dim = feature_dim
        self.point_cloud_type = point_cloud_type
        scenes = {r["scene_id"] for r in self.anno}
        if not scenes <= self.attributes.keys():
            raise ValueError("question scene is missing from attributes")
        self.attributes = {k: v for k, v in self.attributes.items() if k in scenes}
        if self.full_input:
            self.img_feats = torch.load(images, map_location="cpu", weights_only=True)
            self.feats_edge = torch.load(relations, map_location="cpu", weights_only=True)
            for name, bank in (("images", self.img_feats), ("relations", self.feats_edge)):
                if not isinstance(bank, dict) or any(
                    not isinstance(key, str) or not isinstance(value, torch.Tensor)
                    for key, value in bank.items()
                ):
                    raise ValueError(f"{name} must be a tensor-only dictionary with string keys")
            if self.missing_channel_policy == "zero_projected_declared":
                self._validate_declared_missing_channels()
            identities = self.full_manifest["scene_object_ids"]
            for scene, attr in self.attributes.items():
                count = attr["locs"].shape[0]
                expected = [self._integer_object_id(value)
                            for value in attr.get("obj_ids", range(count))]
                saved = identities.get(scene)
                if (not isinstance(saved, list)
                        or any(isinstance(value, bool) or not isinstance(value, int) for value in saved)
                        or saved != expected):
                    raise ValueError(f"full manifest row identities differ from attributes for {scene}")
        (self.scene_feats, self.scene_img_feats, self.scene_masks,
         self.scene_gnn_feats, self.scene_foreground_ids,
         self.scene_radii, self.scene_radius_valid) = self.prepare_scene_features()

    @staticmethod
    def _read_full_manifest(path, inputs):
        if not isinstance(path, (str, Path)):
            raise ValueError("full_manifest must be an explicit JSON path")
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        if (not isinstance(manifest, dict)
                or manifest.get("schema_version") not in ("scalestitch_full_input/v1", "scalestitch_full_input/v2")
                or not isinstance(manifest.get("sha256"), dict)
                or not isinstance(manifest.get("scene_object_ids"), dict)):
            raise ValueError("invalid full input manifest schema")
        if manifest["schema_version"] == "scalestitch_full_input/v2":
            if (manifest.get("missing_channel_policy") != "zero_projected_declared"
                    or not isinstance(manifest.get("missing_channels"), dict)
                    or set(manifest["missing_channels"]) != set(manifest["scene_object_ids"])):
                raise ValueError("v2 requires explicit missing-channel policy and every scene mask")
        elif manifest.get("missing_channel_policy", "reject") != "reject" or "missing_channels" in manifest:
            raise ValueError("v1 cannot enable declared missing channels")
        for name, filename in inputs.items():
            expected = manifest["sha256"].get(name)
            if (not isinstance(expected, str) or len(expected) != 64
                    or any(c not in "0123456789abcdef" for c in expected)):
                raise ValueError(f"full manifest requires a lowercase SHA256 for {name}")
            digest = hashlib.sha256()
            with Path(filename).open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError(f"full input SHA256 mismatch for {name}")
        return manifest

    def _validate_declared_missing_channels(self):
        """Validate every manifest scene in raw source-row order before batching.

        A declared slot must contain an exact all-zero placeholder; every other
        nonself slot must be finite and nonzero. No absent dictionary key is filled.
        """
        for scene, ids in self.full_manifest["scene_object_ids"].items():
            if (not isinstance(ids, list) or not ids or len(ids) > 1000
                    or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids)):
                raise ValueError("invalid declared scene object identities")
            count = len(ids)
            mask = self.full_manifest["missing_channels"][scene]
            if not isinstance(mask, dict) or set(mask) != {"image_rows", "relation_pairs"}:
                raise ValueError("missing-channel mask must specify image_rows and relation_pairs")
            images, pairs = mask["image_rows"], mask["relation_pairs"]
            if (not isinstance(images, list) or any(type(i) is not int or not 0 <= i < count for i in images)
                    or len(set(images)) != len(images)):
                raise ValueError("invalid or repeated missing image source row")
            if (not isinstance(pairs, list) or any(not isinstance(p, list) or len(p) != 2
                    or any(type(i) is not int or not 0 <= i < count for i in p) or p[0] == p[1] for p in pairs)
                    or len({tuple(p) for p in pairs}) != len(pairs)):
                raise ValueError("invalid or repeated missing nonself relation pair")
            actual_images = []
            for row, oid in enumerate(ids):
                feature, _ = self._lookup_object_feature_with_key(
                    self.img_feats, scene, oid, 1024, required=True, allow_zero=True)
                if not torch.any(feature != 0):
                    actual_images.append(row)
            bank = self.feats_edge.get(scene)
            if (bank is None or bank.shape != (count, count, 512) or not bank.is_floating_point()
                    or not torch.isfinite(bank).all()):
                raise ValueError("declared relation bank must be finite floating [N,N,512]")
            zeros = torch.count_nonzero(bank, dim=-1) == 0
            zeros.fill_diagonal_(False)
            actual_pairs = {tuple(p) for p in torch.nonzero(zeros).tolist()}
            if set(actual_images) != set(images) or actual_pairs != {tuple(p) for p in pairs}:
                raise ValueError("raw zero channels differ from the exact declared missing masks")

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        row = self.anno[index]
        if "assigned_ids" not in row:
            raise ValueError("each row must supply its saved object permutation in assigned_ids")
        assignment = self._validate_assigned_ids(
            row["assigned_ids"], self.attributes[row["scene_id"]]["locs"].shape[0]
        )
        (scene_id, feat, image, mask, locs, assignment, gnn,
         foreground, radius, valid) = self.get_anno(index, assignment)
        return (feat, image, mask, locs, int(row.get("obj_id", 0)), assignment,
                gnn, foreground, radius, valid,
                update_caption(row.get("caption", ""), assignment),
                update_caption(row["prompt"], assignment))

    def batches(self):
        return [train_collate_fn([self[i]]) for i in range(len(self))]


def select_rows(rows, question_ids):
    """Select a saved cohort in its recorded order; missing or repeated IDs fail."""
    by_id = {row["qid"]: row for row in rows}
    if len(by_id) != len(rows) or len(set(question_ids)) != len(question_ids):
        raise ValueError("duplicate question IDs")
    return [by_id[qid] for qid in question_ids]
