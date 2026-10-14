# Adapted from Graph3DLLM / Chat-Scene; see LICENSE.host.
import random
import re
from numbers import Integral
import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
IOU_THRESHOLD = 0.99

class BaseDataset(Dataset):

    def __init__(self):
        self.media_type = "point_cloud"
        self.anno = None
        self.attributes = None
        self.feats = None
        self.feats_edge = None
        self.img_feats = None
        self.scene_feats = None
        self.scene_img_feats = None
        self.scene_masks = None
        self.scene_foreground_ids = None
        self.radius_records = None
        self.scene_radii = None
        self.scene_radius_valid = None
        self.radius_contract_mode = "protocol"
        self.radius_epsilon = None
        self.feat_dim = 1024
        self.img_feat_dim = 1024
        self.max_obj_num = 100
        self.knn = 0
        self.point_cloud_type = None
        self.full_input = False

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @staticmethod
    def _object_feature_keys(scan_id, obj_id):
        raw = obj_id.item() if torch.is_tensor(obj_id) else obj_id
        keys = [f"{scan_id}_{raw}"]
        try:
            oid = int(raw)
        except (TypeError, ValueError):
            return keys
        keys.extend(
            [
                f"{scan_id}_{oid}",
                f"{scan_id}_{oid:d}",
                f"{scan_id}_{oid:02d}",
            ]
        )

        seen = set()
        out = []
        for key in keys:
            if key not in seen:
                seen.add(key)
                out.append(key)
        return out

    @classmethod
    def _lookup_object_feature(cls, feats, scan_id, obj_id, dim):
        if feats is None:
            return torch.zeros(dim)
        for key in cls._object_feature_keys(scan_id, obj_id):
            if key in feats:
                return feats[key]
        return torch.zeros(dim)

    @classmethod
    def _lookup_object_feature_with_key(
        cls, feats, scan_id, obj_id, dim, *, required=False, allow_zero=False
    ):
        if feats is not None:
            keys = [key for key in cls._object_feature_keys(scan_id, obj_id)
                    if key in feats]
            if required and len(keys) > 1:
                raise ValueError(f"Ambiguous object feature aliases for {scan_id}: {keys}")
            if keys:
                key = keys[0]
                feature = torch.as_tensor(feats[key])
                if (
                    feature.numel() != dim
                    or not torch.isfinite(feature).all()
                    or (required and (
                        feature.shape != (dim,)
                        or not feature.is_floating_point()
                        or (not allow_zero and not torch.any(feature != 0))
                    ))
                ):
                    raise RuntimeError(
                        f"Invalid object feature for {[scan_id, str(obj_id)]} at {key}"
                    )
                return feature, key
        if required:
            raise KeyError(f"Missing object feature for {[scan_id, str(obj_id)]}")
        return torch.zeros(dim), None

    @staticmethod
    def _integer_object_id(value):
        if torch.is_tensor(value):
            if value.ndim != 0:
                raise ValueError("object IDs must be scalar integers")
            value = value.item()
        if isinstance(value, bool):
            raise ValueError("object IDs must be nonnegative integers")
        if isinstance(value, Integral):
            result = int(value)
        elif isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
            result = int(value)
        else:
            raise ValueError("object IDs must be nonnegative integers")
        if result < 0:
            raise ValueError("object IDs must be nonnegative integers")
        return result

    @classmethod
    def _same_object_id(cls, left, right):
        return cls._integer_object_id(left) == cls._integer_object_id(right)


    def _lookup_object_radius(self, feature_key, scan_id, obj_id):
        if feature_key is not None and feature_key in (self.radius_records or {}):
            record = self.radius_records[feature_key]
            if record["scene_id"] != str(scan_id) or not self._same_object_id(
                record["object_id"], obj_id
            ):
                raise ValueError(
                    "Feature/radius/object identity mismatch at "
                    f"{feature_key}: record={[record['scene_id'], record['object_id']]}, "
                    f"expected={[str(scan_id), str(obj_id)]}"
                )
            raw_rho = torch.as_tensor(record["rho"])
            raw_valid = torch.as_tensor(record["radius_valid"])
            if (raw_rho.ndim != 0 or raw_rho.dtype == torch.bool
                    or raw_rho.is_complex()):
                raise ValueError(f"rho must be a real scalar at {feature_key}")
            if raw_valid.ndim != 0 or raw_valid.dtype != torch.bool:
                raise ValueError(f"radius_valid must be a boolean scalar at {feature_key}")
            rho = raw_rho.to(dtype=torch.float32)
            valid = raw_valid.clone()
            if not torch.isfinite(rho) or rho < 0 or (valid and rho <= 0):
                raise ValueError(f"Invalid rho at {feature_key}: {rho.item()}")
            if self.radius_epsilon is not None and bool(valid) != bool(rho > self.radius_epsilon):
                raise ValueError(f"radius_valid disagrees with rho at {feature_key}")
            return rho, valid
        if self.radius_contract_mode == "protocol":
            raise KeyError(
                f"Missing radius for feature/object {[feature_key, scan_id, str(obj_id)]}"
            )
        return torch.tensor(0.0, dtype=torch.float32), torch.tensor(False)

    def prepare_scene_features(self):
        if self.radius_contract_mode == "protocol":
            # Required scenes come from the questions/attributes, never from
            # available features: otherwise a wholly missing scene is skipped.
            scan_ids = set(self.attributes)
        elif self.feats is not None:
            scan_ids = set('_'.join(x.split('_')[:2]) for x in self.feats.keys())
        else:
            scan_ids = set('_'.join(x.split('_')[:2]) for x in self.img_feats.keys())
        scene_feats = {}
        scene_img_feats = {}
        scene_masks = {}
        scene_radii = {}
        scene_radius_valid = {}
        scene_gnn_feats = {}
        scene_foreground_ids = {}
        for scan_id in tqdm.tqdm(sorted(scan_ids)):


            if scan_id not in self.attributes:
                continue
            scene_attr = self.attributes[scan_id]
            obj_num = scene_attr['locs'].shape[0]
            if self.full_input and (
                scene_attr['locs'].shape != (obj_num, 6)
                or not torch.isfinite(scene_attr['locs']).all()
            ):
                raise ValueError(f"full input locations must be finite [N,6] for {scan_id}")

            obj_ids = scene_attr['obj_ids'] if 'obj_ids' in scene_attr else [_ for _ in range(obj_num)]
            if obj_num == 0 or len(obj_ids) != obj_num:
                raise ValueError(f"Object IDs and locations must be nonempty and aligned for {scan_id}")
            logical_ids = [self._integer_object_id(value) for value in obj_ids]
            if len(set(logical_ids)) != len(logical_ids):
                raise ValueError(f"Duplicate source object IDs for {scan_id}")


            scene_feat = []
            scene_img_feat = []
            scene_mask = []
            scene_radius = []
            radius_valid = []
            for _i, _id in enumerate(obj_ids):
                _id = _id.item() if torch.is_tensor(_id) else _id
                if scan_id == 'scene0217_00':
                    # Offset only the retained pack lookup ID. Never mutate
                    # an obj_ids tensor through a scalar view.
                    _id = logical_ids[_i] + 31
                feature, feature_key = self._lookup_object_feature_with_key(
                    self.feats,
                    scan_id,
                    _id,
                    self.feat_dim,
                    required=self.radius_contract_mode == "protocol",
                )
                rho, is_valid = self._lookup_object_radius(
                    feature_key, scan_id, _id
                )
                scene_feat.append(feature)
                scene_radius.append(rho)
                radius_valid.append(is_valid)
                if self.full_input:
                    # Image IDs are raw source IDs, unlike the retained point
                    # feature/radius pack offset for scene0217_00.
                    image, _ = self._lookup_object_feature_with_key(
                        self.img_feats, scan_id, logical_ids[_i], self.img_feat_dim,
                        required=True, allow_zero=getattr(self, "missing_channel_policy", "reject") == "zero_projected_declared",
                    )
                else:
                    image = self._lookup_object_feature(
                        self.img_feats, scan_id, _id, self.img_feat_dim
                    )
                scene_img_feat.append(image.float())


                scene_mask.append(1)
            filtered_objects = []

            if self.point_cloud_type == "gt":
                scene_foreground_ids[scan_id] = torch.tensor(logical_ids, dtype=torch.long)
            else:

                for _i, obj1 in enumerate(scene_attr["locs"]):
                    keep = True
                    for _j, obj2 in enumerate(scene_attr["locs"]):
                        if _i < _j:
                            box1 = construct_bbox_corners(obj1.tolist()[:3], obj1.tolist()[3:])
                            box2 = construct_bbox_corners(obj2.tolist()[:3], obj2.tolist()[3:])
                            iou = box3d_iou(box1, box2)

                            if iou > IOU_THRESHOLD:
                                keep = False
                                break
                    if keep:
                        filtered_objects.append(_i)
                scene_foreground_ids[scan_id] = torch.LongTensor(filtered_objects)
            scene_feat = torch.stack(scene_feat, dim=0)
            scene_img_feat = torch.stack(scene_img_feat, dim=0)
            scene_mask = torch.tensor(scene_mask, dtype=torch.int)
            scene_radius = torch.stack(scene_radius, dim=0).to(torch.float32)
            radius_valid = torch.stack(radius_valid, dim=0).to(torch.bool)
            scene_feats[scan_id] = scene_feat
            scene_img_feats[scan_id] = scene_img_feat
            scene_masks[scan_id] = scene_mask
            scene_radii[scan_id] = scene_radius
            scene_radius_valid[scan_id] = radius_valid


            gnn_shape = 512
            gnn_feat = []

            if self.full_input:
                if scan_id not in self.feats_edge:
                    raise ValueError(f"Missing full relation bank for {scan_id}")
                gnn_feat = self.feats_edge[scan_id]
                if (gnn_feat.shape != (obj_num, obj_num, gnn_shape)
                        or not gnn_feat.is_floating_point()
                        or not torch.isfinite(gnn_feat).all()):
                    raise ValueError(f"full relation bank must be finite floating [N,N,512] for {scan_id}")
                # Absent pairs remain zero. The host rejects missing *selected*
                # edges after its actual FP32 KNN selection.
            elif self.feats_edge is not None and scan_id in self.feats_edge:
                gnn_feat = self.feats_edge[scan_id]
            else:
                gnn_feat = torch.zeros((len(obj_ids)*self.knn, gnn_shape))
            scene_gnn_feats[scan_id] = gnn_feat

        return (
            scene_feats,
            scene_img_feats,
            scene_masks,
            scene_gnn_feats,
            scene_foreground_ids,
            scene_radii,
            scene_radius_valid,
        )

    def get_anno(self, index, assigned_ids=None):

        scene_id = self.anno[index]["scene_id"]

        if self.attributes is not None:
            scene_attr = self.attributes[scene_id]

            scene_locs = scene_attr["locs"]
        else:
            scene_locs = torch.randn((1, 6))
        scene_feat = self.scene_feats[scene_id]
        if scene_feat.ndim == 1:
            scene_feat = scene_feat.unsqueeze(0)
        scene_img_feat = self.scene_img_feats[scene_id] if self.scene_img_feats is not None else torch.zeros((scene_feat.shape[0], self.img_feat_dim))
        scene_mask = self.scene_masks[scene_id] if self.scene_masks is not None else torch.ones(scene_feat.shape[0], dtype=torch.int)
        if self.scene_radii is None or self.scene_radius_valid is None:
            if self.radius_contract_mode == "protocol":
                raise RuntimeError(f"Protocol radii were not prepared for {scene_id}")
            scene_radius = torch.zeros(scene_feat.shape[0], dtype=torch.float32)
            radius_valid = torch.zeros(scene_feat.shape[0], dtype=torch.bool)
        else:
            scene_radius = self.scene_radii[scene_id]
            radius_valid = self.scene_radius_valid[scene_id]
        expected = scene_feat.shape[0]
        if not (
            scene_locs.shape[0]
            == scene_mask.shape[0]
            == scene_radius.shape[0]
            == radius_valid.shape[0]
            == expected
        ):
            raise RuntimeError(
                f"Feature/radius/location alignment mismatch for {scene_id}: "
                f"feat={expected}, locs={scene_locs.shape[0]}, mask={scene_mask.shape[0]}, "
                f"radius={scene_radius.shape[0]}, valid={radius_valid.shape[0]}"
            )
        if assigned_ids is None:

            assigned_ids = torch.randperm(len(scene_locs))
        else:
            assigned_ids = self._validate_assigned_ids(assigned_ids, len(scene_locs))
        scene_gnn_feat = self.scene_gnn_feats[scene_id]
        scene_foreground_ids = self.scene_foreground_ids[scene_id]
        return (
            scene_id,
            scene_feat,
            scene_img_feat,
            scene_mask,
            scene_locs,
            assigned_ids,
            scene_gnn_feat,
            scene_foreground_ids,
            scene_radius,
            radius_valid,
        )

    @staticmethod
    def _validate_assigned_ids(assigned_ids, obj_num):
        def contains_boolean(value):
            if isinstance(value, (bool, np.bool_)):
                return True
            if torch.is_tensor(value):
                return value.dtype == torch.bool
            if isinstance(value, np.ndarray):
                return value.dtype == np.bool_
            if isinstance(value, (list, tuple)):
                return any(contains_boolean(item) for item in value)
            return False

        # A mixed list such as [True, 0] is inferred as int64 by as_tensor,
        # so inspecting only the converted dtype loses the invalid input type.
        if contains_boolean(assigned_ids):
            raise ValueError("assigned_ids must be an exact integer permutation of scene object rows")
        values = torch.as_tensor(assigned_ids)
        if (
            values.ndim != 1
            or values.dtype == torch.bool
            or values.is_floating_point()
            or values.is_complex()
            or len(values) != obj_num
            or sorted(values.tolist()) != list(range(obj_num))
        ):
            raise ValueError("assigned_ids must be an exact integer permutation of scene object rows")
        return values.to(dtype=torch.long).clone()


def update_caption(caption, assigned_ids):
    new_ids = {int(assigned_id): i for i, assigned_id in enumerate(assigned_ids)}
    id_format = "<OBJ\\d{3}>"
    for match in re.finditer(id_format, caption):
        idx = match.start()
        old_id = int(caption[idx+4:idx+7])
        new_id = int(new_ids[old_id])
        caption = caption[:idx+4] + f"{new_id:03}" + caption[idx+7:]
    return caption


def recover_caption(caption, assigned_ids):
    id_format = "<OBJ\\d{3}>"
    for match in re.finditer(id_format, caption):
        idx = match.start()
        new_id = int(caption[idx+4:idx+7])
        try:
            old_id = int(assigned_ids[new_id])
        except:
            old_id = random.randint(0, len(assigned_ids)-1)
        caption = caption[:idx+4] + f"{old_id:03}" + caption[idx+7:]
    return caption


def get_box3d_min_max(corner):

    min_coord = corner.min(axis=0)
    max_coord = corner.max(axis=0)
    x_min, x_max = min_coord[0], max_coord[0]
    y_min, y_max = min_coord[1], max_coord[1]
    z_min, z_max = min_coord[2], max_coord[2]

    return x_min, x_max, y_min, y_max, z_min, z_max


def box3d_iou(corners1, corners2):

    x_min_1, x_max_1, y_min_1, y_max_1, z_min_1, z_max_1 = get_box3d_min_max(corners1)

    x_min_2, x_max_2, y_min_2, y_max_2, z_min_2, z_max_2 = get_box3d_min_max(corners2)

    xA = np.maximum(x_min_1, x_min_2)
    yA = np.maximum(y_min_1, y_min_2)
    zA = np.maximum(z_min_1, z_min_2)
    xB = np.minimum(x_max_1, x_max_2)
    yB = np.minimum(y_max_1, y_max_2)
    zB = np.minimum(z_max_1, z_max_2)
    inter_vol = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0) * np.maximum((zB - zA), 0)
    box_vol_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1) * (z_max_1 - z_min_1)

    box_vol_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2) * (z_max_2 - z_min_2)


    iou = inter_vol / (box_vol_1 + box_vol_2 - inter_vol + 1e-8)


    return iou


def construct_bbox_corners(center, box_size):
    sx, sy, sz = box_size
    x_corners = [sx / 2, sx / 2, -sx / 2, -sx / 2, sx / 2, sx / 2, -sx / 2, -sx / 2]
    y_corners = [sy / 2, -sy / 2, -sy / 2, sy / 2, sy / 2, -sy / 2, -sy / 2, sy / 2]
    z_corners = [sz / 2, sz / 2, sz / 2, sz / 2, -sz / 2, -sz / 2, -sz / 2, -sz / 2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)

    return corners_3d


def train_collate_fn(batch):
    (
        scene_feats,
        scene_img_feats,
        scene_masks,
        scene_locs,
        obj_ids,
        assigned_ids,
        scene_gnn_feats,
        scene_foreground_ids,
        scene_radii,
        radius_valids,
        captions,
        questions,
    ) = zip(*batch)
    batch_scene_feat = pad_sequence(scene_feats, batch_first=True)
    batch_scene_img_feat = pad_sequence(scene_img_feats, batch_first=True)
    batch_scene_mask = pad_sequence(scene_masks, batch_first=True).to(torch.bool)
    batch_scene_locs = pad_sequence(scene_locs, batch_first=True)
    batch_assigned_ids = pad_sequence(assigned_ids, batch_first=True)
    if any(edges.ndim == 3 for edges in scene_gnn_feats):
        if len(batch) != 1 or scene_gnn_feats[0].ndim != 3:
            raise ValueError("dense full relation banks require batch size one; pair axes cannot be padded")
        batch_scene_gnn_feats = scene_gnn_feats[0].unsqueeze(0)
    else:
        batch_scene_gnn_feats = pad_sequence(scene_gnn_feats, batch_first=True)
    batch_scene_radius = pad_sequence(scene_radii, batch_first=True)
    batch_radius_valid = pad_sequence(radius_valids, batch_first=True).to(torch.bool)


    obj_ids = torch.tensor(obj_ids)
    foreground_ids = scene_foreground_ids
    return {
        "scene_feat": batch_scene_feat,
        "scene_img_feat": batch_scene_img_feat,
        "scene_locs": batch_scene_locs,
        "scene_mask": batch_scene_mask,
        "assigned_ids": batch_assigned_ids,
        "scene_gnn_feats": batch_scene_gnn_feats,
        "scene_radius": batch_scene_radius,
        "radius_valid": batch_radius_valid,
        "foreground_ids": foreground_ids,

        "obj_ids": obj_ids,
        "answers": captions,
        "questions": questions


    }
