import csv
import json
import os
import random

import numpy as np
import numpy.ma as ma
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


LINEMOD_OBJLIST = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
LMO_OBJLIST = [1, 5, 6, 8, 9, 10, 11, 12]
SYMMETRIC_LINEMOD_CLASS_INDICES = [7, 8]  # eggbox, glue in LINEMOD_OBJLIST.


class PoseDataset(data.Dataset):
    """BOP-format Occlusion LINEMOD / LM-O evaluation dataset.

    The returned tensors follow the same contract as datasets/linemod/dataset.py:
    points, choose, img, target, sampled_model_points, linemod_class_index.
    """

    def __init__(
        self,
        mode,
        num,
        add_noise,
        root,
        noise_trans,
        refine,
        objlist=None,
        use_targets=True,
        models_dir="models_eval",
        detector_results="",
        detector_score_threshold=0.0,
        detector_match="score",
        detector_fallback_to_gt=False,
        sample_seed=0,
    ):
        if mode not in ("eval", "test", "oracle_train"):
            raise ValueError("Occlusion LINEMOD loader supports eval/test/oracle_train only.")

        self.mode = mode
        self.num = num
        self.add_noise = add_noise
        self.root = root
        self.noise_trans = noise_trans
        self.refine = refine
        self.objlist = list(objlist or LMO_OBJLIST)
        self.linemod_objlist = LINEMOD_OBJLIST
        self.models_dir = models_dir
        self.use_targets = use_targets
        self.detector_results = detector_results
        self.detector_score_threshold = detector_score_threshold
        self.detector_match = detector_match
        self.detector_fallback_to_gt = detector_fallback_to_gt
        self.sample_seed = int(sample_seed)
        self.detections_by_image = (
            load_detector_results(detector_results, self.objlist)
            if detector_results
            else {}
        )

        self.num_pt_mesh_large = 500
        self.num_pt_mesh_small = 500
        self.symmetry_obj_idx = SYMMETRIC_LINEMOD_CLASS_INDICES
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        with open(os.path.join(root, "camera.json"), "r") as f:
            camera = json.load(f)
        self.width = int(camera.get("width", 640))
        self.height = int(camera.get("height", 480))
        self.default_cam = {
            "fx": float(camera["fx"]),
            "fy": float(camera["fy"]),
            "cx": float(camera["cx"]),
            "cy": float(camera["cy"]),
            "depth_scale": float(camera.get("depth_scale", 1.0)),
        }
        # Match the historical DenseFusion variable convention:
        # xmap stores row/v coordinates and ymap stores column/u coordinates.
        self.xmap = np.array([[i for _ in range(self.width)] for i in range(self.height)])
        self.ymap = np.array([[j for j in range(self.width)] for _ in range(self.height)])

        self.model_points = {}
        self.full_model_points = {}
        model_root = os.path.join(root, models_dir)
        for obj_id in self.objlist:
            points = ply_vtx(os.path.join(model_root, "obj_{0:06d}.ply".format(obj_id))) / 1000.0
            self.full_model_points[obj_id] = points.astype(np.float32)
            self.model_points[obj_id] = points.astype(np.float32)

        with open(os.path.join(model_root, "models_info.json"), "r") as f:
            models_info = json.load(f)
        self.diameters = {
            int(obj_id): float(info["diameter"]) / 1000.0
            for obj_id, info in models_info.items()
            if int(obj_id) in self.objlist
        }

        self.samples = self._build_samples()
        self.length = len(self.samples)

    def _build_samples(self):
        test_root = os.path.join(self.root, "test")
        scene_ids = [
            int(name)
            for name in os.listdir(test_root)
            if os.path.isdir(os.path.join(test_root, name))
        ]
        scene_ids.sort()

        target_counts = {}
        target_path = os.path.join(self.root, "test_targets_bop19.json")
        if self.use_targets and os.path.exists(target_path):
            with open(target_path, "r") as f:
                targets = json.load(f)
            for target in targets:
                obj_id = int(target["obj_id"])
                if obj_id not in self.objlist:
                    continue
                key = (int(target["scene_id"]), int(target["im_id"]), obj_id)
                target_counts[key] = int(target.get("inst_count", 1))

        samples = []
        for scene_id in scene_ids:
            scene_dir = os.path.join(test_root, "{0:06d}".format(scene_id))
            with open(os.path.join(scene_dir, "scene_gt.json"), "r") as f:
                scene_gt = json.load(f)
            with open(os.path.join(scene_dir, "scene_gt_info.json"), "r") as f:
                scene_gt_info = json.load(f)
            with open(os.path.join(scene_dir, "scene_camera.json"), "r") as f:
                scene_camera = json.load(f)

            for im_key in sorted(scene_gt.keys(), key=lambda item: int(item)):
                im_id = int(im_key)
                per_object_count = {}
                for gt_id, meta in enumerate(scene_gt[im_key]):
                    obj_id = int(meta["obj_id"])
                    if obj_id not in self.objlist:
                        continue
                    key = (scene_id, im_id, obj_id)
                    if target_counts:
                        used = per_object_count.get(obj_id, 0)
                        if used >= target_counts.get(key, 0):
                            continue
                        per_object_count[obj_id] = used + 1

                    info = scene_gt_info[im_key][gt_id]
                    samples.append(
                        {
                            "scene_id": scene_id,
                            "im_id": im_id,
                            "gt_id": gt_id,
                            "obj_id": obj_id,
                            "gt": meta,
                            "gt_info": info,
                            "camera": scene_camera[im_key],
                            "rgb_path": os.path.join(scene_dir, "rgb", "{0:06d}.png".format(im_id)),
                            "depth_path": os.path.join(scene_dir, "depth", "{0:06d}.png".format(im_id)),
                            "mask_path": os.path.join(
                                scene_dir,
                                "mask_visib",
                                "{0:06d}_{1:06d}.png".format(im_id, gt_id),
                            ),
                        }
                    )
        return samples

    def __getitem__(self, index):
        sample = self.samples[index]
        img = Image.open(sample["rgb_path"]).convert("RGB")
        depth = np.array(Image.open(sample["depth_path"]))
        detection = self.select_detection(sample)
        using_detector = self.detector_results and detection is not None
        if self.detector_results and detection is None and not self.detector_fallback_to_gt:
            cc = torch.LongTensor([0])
            return cc, cc, cc, cc, cc, cc

        if using_detector:
            mask_label = bbox_to_mask(detection["bbox"], self.height, self.width)
            bbox = detection["bbox"]
        else:
            mask_label = np.array(Image.open(sample["mask_path"])) > 0
            bbox = sample["gt_info"].get("bbox_visib") or mask_to_bbox(mask_label)

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        mask = mask_label * mask_depth

        if self.add_noise:
            img = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)(img)

        if mask.sum() == 0:
            cc = torch.LongTensor([0])
            return cc, cc, cc, cc, cc, cc

        rmin, rmax, cmin, cmax = get_bbox(bbox, self.height, self.width)

        img_np = np.array(img)[:, :, :3]
        img_np = np.transpose(img_np, (2, 0, 1))
        img_masked = img_np[:, rmin:rmax, cmin:cmax]

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) == 0:
            cc = torch.LongTensor([0])
            return cc, cc, cc, cc, cc, cc

        rng = np.random.RandomState(index + 1000003 * self.sample_seed)
        if len(choose) > self.num:
            choose = rng.choice(choose, self.num, replace=False)
        else:
            choose = np.pad(choose, (0, self.num - len(choose)), "wrap")

        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        cam = self._camera(sample)
        pt2 = depth_masked * cam["depth_scale"]
        pt0 = (ymap_masked - cam["cx"]) * pt2 / cam["fx"]
        pt1 = (xmap_masked - cam["cy"]) * pt2 / cam["fy"]
        cloud = np.concatenate((pt0, pt1, pt2), axis=1) / 1000.0

        obj_id = sample["obj_id"]
        model_points = self.model_points[obj_id]
        if len(model_points) > self.num_pt_mesh_small:
            keep = rng.choice(len(model_points), self.num_pt_mesh_small, replace=False)
            model_points_sampled = model_points[keep]
        else:
            keep = rng.choice(len(model_points), self.num_pt_mesh_small, replace=True)
            model_points_sampled = model_points[keep]

        target_r = np.resize(np.array(sample["gt"]["cam_R_m2c"], dtype=np.float32), (3, 3))
        target_t = np.array(sample["gt"]["cam_t_m2c"], dtype=np.float32) / 1000.0
        target = np.dot(model_points_sampled, target_r.T) + target_t
        linemod_idx = self.linemod_objlist.index(obj_id)

        return (
            torch.from_numpy(cloud.astype(np.float32)),
            torch.LongTensor(choose.astype(np.int32)),
            self.norm(torch.from_numpy(img_masked.astype(np.float32))),
            torch.from_numpy(target.astype(np.float32)),
            torch.from_numpy(model_points_sampled.astype(np.float32)),
            torch.LongTensor([linemod_idx]),
        )

    def _camera(self, sample):
        cam_k = sample["camera"].get("cam_K")
        if cam_k:
            return {
                "fx": float(cam_k[0]),
                "fy": float(cam_k[4]),
                "cx": float(cam_k[2]),
                "cy": float(cam_k[5]),
                "depth_scale": float(sample["camera"].get("depth_scale", self.default_cam["depth_scale"])),
            }
        return self.default_cam

    def get_sample_meta(self, index):
        return self.samples[index]

    def select_detection(self, sample):
        if not self.detector_results:
            return None

        key = (int(sample["scene_id"]), int(sample["im_id"]))
        candidates = [
            det
            for det in self.detections_by_image.get(key, [])
            if int(det["obj_id"]) == int(sample["obj_id"])
            and float(det.get("score", 1.0)) >= self.detector_score_threshold
        ]
        if not candidates:
            return None

        if self.detector_match == "iou":
            gt_bbox = sample["gt_info"].get("bbox_visib") or [0, 0, 0, 0]
            return max(
                candidates,
                key=lambda det: (bbox_iou(det["bbox"], gt_bbox), float(det.get("score", 1.0))),
            )
        return max(candidates, key=lambda det: float(det.get("score", 1.0)))

    def get_input_source(self, index):
        sample = self.samples[index]
        detection = self.select_detection(sample)
        if self.detector_results and detection is not None:
            return "detector_bbox_depth", detection
        if self.detector_results and detection is None and self.detector_fallback_to_gt:
            return "gt_visible_mask_fallback", None
        return "gt_visible_mask", None

    def get_model_points(self, obj_id):
        return self.full_model_points[int(obj_id)]

    def get_diameter(self, obj_id):
        return self.diameters[int(obj_id)]

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        return self.num_pt_mesh_large if self.refine else self.num_pt_mesh_small

    def __len__(self):
        return self.length


def mask_to_bbox(mask):
    mask = mask.astype(np.uint8)
    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        return [0, 0, 0, 0]
    x_min, x_max = int(cols.min()), int(cols.max())
    y_min, y_max = int(rows.min()), int(rows.max())
    return [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]


def bbox_to_mask(bbox, height, width):
    x, y, w, h = [float(v) for v in bbox]
    x0 = max(0, int(round(x)))
    y0 = max(0, int(round(y)))
    x1 = min(width, int(round(x + w)))
    y1 = min(height, int(round(y + h)))
    mask = np.zeros((height, width), dtype=np.bool_)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def bbox_iou(left, right):
    lx, ly, lw, lh = [float(v) for v in left]
    rx, ry, rw, rh = [float(v) for v in right]
    lx2, ly2 = lx + lw, ly + lh
    rx2, ry2 = rx + rw, ry + rh
    ix1, iy1 = max(lx, rx), max(ly, ry)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, lw) * max(0.0, lh) + max(0.0, rw) * max(0.0, rh) - inter
    return inter / union if union > 0 else 0.0


def load_detector_results(path, objlist):
    if not os.path.exists(path):
        raise FileNotFoundError("Detector results file not found: {0}".format(path))

    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            raw = raw.get("detections", raw.get("results", []))
        rows = raw
    else:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))

    objset = {int(item) for item in objlist}
    grouped = {}
    for row in rows:
        det = normalize_detection_row(row)
        if det is None or det["obj_id"] not in objset:
            continue
        key = (det["scene_id"], det["im_id"])
        grouped.setdefault(key, []).append(det)

    for detections in grouped.values():
        detections.sort(key=lambda det: float(det.get("score", 1.0)), reverse=True)
    return grouped


def normalize_detection_row(row):
    try:
        scene_id = int(row.get("scene_id", row.get("scene", row.get("sceneId"))))
        im_id = int(row.get("im_id", row.get("image_id", row.get("imId"))))
        obj_id = int(row.get("obj_id", row.get("class_id", row.get("category_id"))))
    except (TypeError, ValueError):
        return None

    bbox = parse_bbox(row)
    if bbox is None:
        return None

    score = row.get("score", row.get("confidence", row.get("conf", 1.0)))
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 1.0

    return {
        "scene_id": scene_id,
        "im_id": im_id,
        "obj_id": obj_id,
        "score": score,
        "bbox": bbox,
    }


def parse_bbox(row):
    for key in ("bbox_est", "bbox", "box"):
        value = row.get(key)
        if value is not None:
            if isinstance(value, str):
                value = value.replace("[", "").replace("]", "").replace(",", " ").split()
            if isinstance(value, (list, tuple)) and len(value) >= 4:
                try:
                    return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
                except (TypeError, ValueError):
                    return None

    xywh_keys = ("x", "y", "w", "h")
    if all(key in row for key in xywh_keys):
        try:
            return [float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"])]
        except (TypeError, ValueError):
            return None

    xyxy_keys = ("x1", "y1", "x2", "y2")
    if all(key in row for key in xyxy_keys):
        try:
            x1, y1, x2, y2 = [float(row[key]) for key in xyxy_keys]
            return [x1, y1, x2 - x1, y2 - y1]
        except (TypeError, ValueError):
            return None

    return None


def get_bbox(bbox, height=480, width=640):
    border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    bbx[0] = max(0, bbx[0])
    bbx[1] = min(height - 1, bbx[1])
    bbx[2] = max(0, bbx[2])
    bbx[3] = min(width - 1, bbx[3])
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = _snap_border(rmax - rmin, border_list)
    c_b = _snap_border(cmax - cmin, border_list)
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        rmax -= rmin
        rmin = 0
    if cmin < 0:
        cmax -= cmin
        cmin = 0
    if rmax > height:
        rmin -= rmax - height
        rmax = height
    if cmax > width:
        cmin -= cmax - width
        cmax = width
    return max(0, rmin), min(height, rmax), max(0, cmin), min(width, cmax)


def _snap_border(value, border_list):
    for idx in range(len(border_list) - 1):
        if value > border_list[idx] and value < border_list[idx + 1]:
            return border_list[idx + 1]
    return value


def ply_vtx(path):
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Invalid PLY file without end_header: {0}".format(path))
            decoded = line.decode("ascii", errors="ignore").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break

        if header_lines[0] != "ply":
            raise ValueError("Invalid PLY file: {0}".format(path))

        fmt = ""
        n_vertices = 0
        vertex_props = []
        in_vertex = False
        for line in header_lines:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[:2] == ["element", "vertex"]:
                n_vertices = int(parts[2])
                in_vertex = True
            elif parts[0] == "element" and parts[1] != "vertex":
                in_vertex = False
            elif in_vertex and parts[0] == "property" and parts[1] != "list":
                vertex_props.append((parts[2], parts[1]))

        if fmt == "ascii":
            pts = []
            for _ in range(n_vertices):
                pts.append(np.float32(f.readline().decode("ascii").split()[:3]))
            return np.array(pts)

        if fmt != "binary_little_endian":
            raise ValueError("Unsupported PLY format {0}: {1}".format(fmt, path))

        dtype_map = {
            "char": "i1",
            "uchar": "u1",
            "int8": "i1",
            "uint8": "u1",
            "short": "<i2",
            "ushort": "<u2",
            "int16": "<i2",
            "uint16": "<u2",
            "int": "<i4",
            "uint": "<u4",
            "int32": "<i4",
            "uint32": "<u4",
            "float": "<f4",
            "float32": "<f4",
            "double": "<f8",
            "float64": "<f8",
        }
        dtype = np.dtype([(name, dtype_map[prop_type]) for name, prop_type in vertex_props])
        vertices = np.fromfile(f, dtype=dtype, count=n_vertices)
        return np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
