#!/usr/bin/env python3
"""Export failure-case visualizations without retraining.

The script first tries to reuse existing visualization outputs or evaluation
logs. If usable files are not found, it can run RAFusion inference with
provided checkpoints and a LINEMOD/Occlusion-LINEMOD style dataset root. When
neither checkpoints nor prediction outputs are available, it writes an explicit
status file and does not fabricate ``failure_cases.png``.
"""

from __future__ import print_function

import argparse
import copy
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


LINEMOD_OBJLIST = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
LINEMOD_NAMES = {
    1: "ape",
    2: "benchvise",
    4: "camera",
    5: "can",
    6: "cat",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone",
}
PREFERRED_FAILURE_OBJECTS = [6, 10, 1]  # cat, eggbox, ape
SYMMETRIC_OBJ_IDS = [10, 11]
BBOX_EDGES = [
    (0, 1),
    (1, 3),
    (3, 2),
    (2, 0),
    (4, 5),
    (5, 7),
    (7, 6),
    (6, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize RAFusion/DenseFusion failure cases without training."
    )
    parser.add_argument("--dataset", default="linemod", choices=["linemod"])
    parser.add_argument(
        "--dataset_root",
        default="datasets/linemod/Linemod_preprocessed",
        help="LINEMOD or Occlusion-LINEMOD style dataset root.",
    )
    parser.add_argument("--model", default="", help="PoseNet checkpoint.")
    parser.add_argument("--refine_model", default="", help="PoseRefineNet checkpoint.")
    parser.add_argument(
        "--prediction_dir",
        default="",
        help="Optional directory with existing prediction/visualization outputs.",
    )
    parser.add_argument(
        "--failure_log",
        default="experiments/eval_result/linemod/eval_result_logs.txt",
        help="Evaluation log used to prioritize known failed samples.",
    )
    parser.add_argument("--output_dir", default="revision_outputs")
    parser.add_argument("--output_name", default="failure_cases.png")
    parser.add_argument("--num_cases", type=int, default=4)
    parser.add_argument("--max_eval_samples", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--num_points", type=int, default=500)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def status_paths(output_dir):
    return (
        os.path.join(output_dir, "failure_cases_status.txt"),
        os.path.join(output_dir, "failure_cases_metadata.json"),
    )


def write_status(output_dir, status, missing=None, sources=None):
    ensure_dir(output_dir)
    missing = missing or []
    sources = sources or []
    txt_path, json_path = status_paths(output_dir)
    lines = [
        "failure_cases.png was not generated.",
        "Status: {0}".format(status),
    ]
    if missing:
        lines.append("Missing files/resources:")
        lines.extend(["- {0}".format(item) for item in missing])
    if sources:
        lines.append("Checked sources:")
        lines.extend(["- {0}".format(item) for item in sources])
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(json_path, "w") as f:
        json.dump({"status": status, "missing": missing, "checked_sources": sources}, f, indent=2)
    print("\n".join(lines))


def find_first(patterns):
    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern, recursive=True))
    matches = [path for path in matches if os.path.isfile(path)]
    return sorted(matches)[0] if matches else ""


def auto_find_checkpoints(args):
    model = args.model if args.model and os.path.exists(args.model) else ""
    refine = args.refine_model if args.refine_model and os.path.exists(args.refine_model) else ""

    if not model:
        model = find_first(
            [
                "trained_checkpoints/**/pose_model*.pth",
                "trained_models/**/pose_model*.pth",
            ]
        )
    if not refine:
        refine = find_first(
            [
                "trained_checkpoints/**/pose_refine_model*.pth",
                "trained_models/**/pose_refine_model*.pth",
            ]
        )
    return model, refine


def candidate_search_dirs(args):
    dirs = []
    if args.prediction_dir:
        dirs.append(args.prediction_dir)
    dirs.extend(
        [
            "revision_outputs",
            "experiments/eval_result",
            "experiments",
            "output",
        ]
    )
    return [path for path in dirs if path and os.path.exists(path)]


def parse_failure_log(log_path):
    failures = {}
    if not os.path.exists(log_path):
        return failures
    pattern = re.compile(r"No\.\s*(\d+)\s+NOT Pass!\s+Distance:\s*([0-9.eE+-]+)")
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                failures[int(match.group(1))] = float(match.group(2))
    return failures


def find_existing_visualization_cases(args):
    """Find existing visualizations and rank them with eval logs when possible."""

    ranked_cases = []
    checked = []
    for root in candidate_search_dirs(args):
        checked.append(root)
        log_failures = {}
        for log_path in glob.glob(os.path.join(root, "**", "eval_result_logs.txt"), recursive=True):
            log_failures.update(parse_failure_log(log_path))

        images = []
        for pattern in (
            "**/visualization_*.png",
            "**/*failure*.png",
            "**/*failure*.jpg",
            "**/*failure*.jpeg",
        ):
            images.extend(glob.glob(os.path.join(root, pattern), recursive=True))

        for image_path in sorted(set(images)):
            name = os.path.basename(image_path)
            index_match = re.search(r"visualization_(\d+)", name)
            score = 0.0
            if index_match:
                score = log_failures.get(int(index_match.group(1)), 0.0)
            ranked_cases.append(
                {
                    "image_path": image_path,
                    "score": score,
                    "object_name": "existing output",
                    "reason": "selected from existing visualization output",
                }
            )

    ranked_cases.sort(key=lambda item: item["score"], reverse=True)
    return ranked_cases, checked


def tile_existing_cases(cases, output_path, num_cases):
    from PIL import Image, ImageDraw, ImageFont

    selected = cases[:num_cases]
    if not selected:
        return False

    tile_w, tile_h, caption_h = 320, 240, 72
    canvas = Image.new("RGB", (tile_w * len(selected), tile_h + caption_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for col, case in enumerate(selected):
        img = Image.open(case["image_path"]).convert("RGB")
        img.thumbnail((tile_w, tile_h))
        x0 = col * tile_w + (tile_w - img.width) // 2
        canvas.paste(img, (x0, 0))
        caption = "{0}\n{1}".format(case["object_name"], case["reason"])
        draw.multiline_text((col * tile_w + 8, tile_h + 8), caption, fill=(0, 0, 0), font=font)

    canvas.save(output_path)
    return True


def dataset_available(root):
    checks = [
        root,
        os.path.join(root, "data"),
        os.path.join(root, "models"),
    ]
    return all(os.path.exists(path) for path in checks)


def select_device(args):
    if args.device == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else None
    if args.device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_state_dict_strict(model, checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if isinstance(state, dict):
        cleaned = {}
        for key, value in state.items():
            if key.startswith("module."):
                key = key[len("module.") :]
            cleaned[key] = value
        state = cleaned
    model.load_state_dict(state)


def quaternion_to_matrix(quaternion):
    from lib.transformations import quaternion_matrix

    return quaternion_matrix(quaternion)[:3, :3]


def refine_prediction(refiner, points, emb, idx, my_r, my_t, iterations, num_points, device):
    from lib.transformations import quaternion_from_matrix, quaternion_matrix

    for _ in range(iterations):
        t_tensor = (
            torch.from_numpy(my_t.astype(np.float32))
            .to(device)
            .view(1, 3)
            .repeat(num_points, 1)
            .contiguous()
            .view(1, num_points, 3)
        )
        current_matrix = quaternion_matrix(my_r)
        r_tensor = (
            torch.from_numpy(current_matrix[:3, :3].astype(np.float32))
            .to(device)
            .view(1, 3, 3)
        )
        current_matrix[0:3, 3] = my_t

        new_points = torch.bmm((points - t_tensor), r_tensor).contiguous()
        pred_r, pred_t = refiner(new_points, emb, idx)
        pred_r = pred_r.view(1, 1, -1)
        pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, 1, 1)
        delta_r = pred_r.view(-1).detach().cpu().numpy()
        delta_t = pred_t.view(-1).detach().cpu().numpy()

        delta_matrix = quaternion_matrix(delta_r)
        delta_matrix[0:3, 3] = delta_t
        final_matrix = np.dot(current_matrix, delta_matrix)
        final_r = copy.deepcopy(final_matrix)
        final_r[0:3, 3] = 0
        my_r = quaternion_from_matrix(final_r, True)
        my_t = np.array([final_matrix[0, 3], final_matrix[1, 3], final_matrix[2, 3]])
    return my_r, my_t


def compute_add_distance(model_points, target, pred_r, pred_t, obj_id):
    pred_rot = quaternion_to_matrix(pred_r)
    pred = np.dot(model_points, pred_rot.T) + pred_t
    if obj_id in SYMMETRIC_OBJ_IDS:
        diff = pred[:, None, :] - target[None, :, :]
        nearest = np.min(np.linalg.norm(diff, axis=2), axis=1)
        return float(np.mean(nearest))
    return float(np.mean(np.linalg.norm(pred - target, axis=1)))


def load_linemod_thresholds(config_dir):
    try:
        import yaml

        path = os.path.join(config_dir, "models_info.yml")
        with open(path, "r") as f:
            meta = yaml.load(f, Loader=yaml.FullLoader)
        return {obj_id: meta[obj_id]["diameter"] / 1000.0 * 0.1 for obj_id in meta}
    except Exception:
        return {}


def get_linemod_meta(dataset, index):
    obj_id = dataset.list_obj[index]
    rank = dataset.list_rank[index]
    if obj_id == 2:
        for meta in dataset.meta[obj_id][rank]:
            if meta["obj_id"] == 2:
                return meta
    return dataset.meta[obj_id][rank][0]


def bbox_corners(points):
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    return np.array(
        [
            [mins[0], mins[1], mins[2]],
            [maxs[0], mins[1], mins[2]],
            [mins[0], maxs[1], mins[2]],
            [maxs[0], maxs[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [maxs[0], mins[1], maxs[2]],
            [mins[0], maxs[1], maxs[2]],
            [maxs[0], maxs[1], maxs[2]],
        ],
        dtype=np.float32,
    )


def project_points(points, camera):
    cx, cy, fx, fy = camera
    z = np.maximum(points[:, 2], 1e-6)
    u = points[:, 0] * fx / z + cx
    v = points[:, 1] * fy / z + cy
    return np.stack([u, v], axis=1).astype(np.int32)


def draw_bbox(image, pixels, color):
    import cv2

    for start, end in BBOX_EDGES:
        p0 = tuple(int(x) for x in pixels[start])
        p1 = tuple(int(x) for x in pixels[end])
        cv2.line(image, p0, p1, color=color, thickness=2, lineType=cv2.LINE_AA)


def infer_failure_reason(dataset, index, obj_id):
    try:
        import cv2

        depth = cv2.imread(dataset.list_depth[index], cv2.IMREAD_UNCHANGED)
        label = cv2.imread(dataset.list_label[index], cv2.IMREAD_UNCHANGED)
        if depth is not None and label is not None:
            if label.ndim == 3:
                mask = label[:, :, 0] > 0
            else:
                mask = label > 0
            visible_pixels = int(mask.sum())
            valid_depth = int(((depth > 0) & mask).sum())
            depth_ratio = valid_depth / float(max(visible_pixels, 1))
            if visible_pixels < 1200:
                return "severe occlusion"
            if depth_ratio < 0.65:
                return "depth missing"
    except Exception:
        pass

    if obj_id in SYMMETRIC_OBJ_IDS:
        return "symmetric ambiguity"
    if obj_id in (1, 6):
        return "severe occlusion"
    return "ambiguous geometry"


def render_failure_case(dataset, index, pred_r, pred_t, distance, threshold):
    import cv2

    obj_id = dataset.list_obj[index]
    object_name = LINEMOD_NAMES.get(obj_id, "object_{0}".format(obj_id))
    image = cv2.imread(dataset.list_rgb[index], cv2.IMREAD_COLOR)
    if image is None:
        return None

    meta = get_linemod_meta(dataset, index)
    target_r = np.resize(np.array(meta["cam_R_m2c"]), (3, 3))
    target_t = np.array(meta["cam_t_m2c"], dtype=np.float32) / 1000.0

    model_points = dataset.pt[obj_id] / 1000.0
    corners = bbox_corners(model_points)

    gt_corners = np.dot(corners, target_r.T) + target_t
    pred_corners = np.dot(corners, quaternion_to_matrix(pred_r).T) + pred_t

    camera = [dataset.cam_cx, dataset.cam_cy, dataset.cam_fx, dataset.cam_fy]
    gt_pixels = project_points(gt_corners, camera)
    pred_pixels = project_points(pred_corners, camera)

    draw_bbox(image, gt_pixels, color=(0, 255, 0))  # green: GT
    draw_bbox(image, pred_pixels, color=(255, 0, 0))  # blue: RAFusion prediction

    reason = infer_failure_reason(dataset, index, obj_id)
    ratio = distance / threshold if threshold else 0.0
    return {
        "image": image[:, :, ::-1],  # BGR to RGB
        "object_id": obj_id,
        "object_name": object_name,
        "reason": reason,
        "distance": distance,
        "threshold": threshold,
        "failure_ratio": ratio,
        "index": index,
    }


def compose_failure_grid(cases, output_path):
    from PIL import Image, ImageDraw, ImageFont

    if not cases:
        return False

    tile_w, tile_h, caption_h = 340, 255, 82
    canvas = Image.new("RGB", (tile_w * len(cases), tile_h + caption_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for col, case in enumerate(cases):
        img = Image.fromarray(case["image"]).convert("RGB")
        img.thumbnail((tile_w, tile_h))
        x0 = col * tile_w + (tile_w - img.width) // 2
        canvas.paste(img, (x0, 0))
        caption = "{0} | {1}\nADD={2:.4f} m, thr={3:.4f} m".format(
            case["object_name"],
            case["reason"],
            case["distance"],
            case["threshold"],
        )
        draw.multiline_text((col * tile_w + 8, tile_h + 8), caption, fill=(0, 0, 0), font=font)

    canvas.save(output_path)
    return True


def run_linemod_inference(args, model_path, refine_model_path, output_path):
    from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
    from lib.network import PoseNet, PoseRefineNet

    device = select_device(args)
    if device is None:
        raise RuntimeError("CUDA was requested but is not available.")

    estimator = PoseNet(num_points=args.num_points, num_obj=len(LINEMOD_OBJLIST)).to(device)
    refiner = PoseRefineNet(num_points=args.num_points, num_obj=len(LINEMOD_OBJLIST)).to(device)
    load_state_dict_strict(estimator, model_path, device)
    load_state_dict_strict(refiner, refine_model_path, device)
    estimator.eval()
    refiner.eval()

    dataset = PoseDataset_linemod("eval", args.num_points, False, args.dataset_root, 0.0, True)
    thresholds = load_linemod_thresholds("datasets/linemod/dataset_config")

    log_failures = parse_failure_log(args.failure_log) if args.failure_log else {}
    known_failures = []
    for index, distance in log_failures.items():
        if 0 <= index < len(dataset):
            known_failures.append((index, distance))
    known_failures.sort(
        key=lambda item: (
            dataset.list_obj[item[0]] in PREFERRED_FAILURE_OBJECTS,
            item[1],
        ),
        reverse=True,
    )

    preferred = []
    other = []
    for idx, obj_id in enumerate(dataset.list_obj):
        if obj_id in PREFERRED_FAILURE_OBJECTS:
            preferred.append(idx)
        else:
            other.append(idx)
    eval_indices = []
    for index, _ in known_failures:
        if index not in eval_indices:
            eval_indices.append(index)
    for index in preferred + other:
        if len(eval_indices) >= args.max_eval_samples:
            break
        if index not in eval_indices:
            eval_indices.append(index)
    eval_indices = eval_indices[: args.max_eval_samples]

    candidates = []
    with torch.no_grad():
        for index in eval_indices:
            sample = dataset[index]
            points, choose, img, target, model_points, obj_idx = sample
            if not hasattr(points, "dim") or points.dim() != 2:
                continue

            obj_id = dataset.list_obj[index]
            points = points.unsqueeze(0).to(device)
            choose = choose.unsqueeze(0).to(device)
            img = img.unsqueeze(0).to(device)
            target_tensor = target.unsqueeze(0).to(device)
            model_points_tensor = model_points.unsqueeze(0).to(device)
            obj_idx = obj_idx.unsqueeze(0).to(device)

            pred_r, pred_t, pred_c, emb = estimator(img, points, choose, obj_idx)
            pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, args.num_points, 1)
            pred_c = pred_c.view(1, args.num_points)
            _, which_max = torch.max(pred_c, 1)
            pred_t = pred_t.view(args.num_points, 1, 3)

            my_r = pred_r[0][which_max[0]].view(-1).detach().cpu().numpy()
            my_t = (
                points.view(args.num_points, 1, 3) + pred_t
            )[which_max[0]].view(-1).detach().cpu().numpy()

            my_r, my_t = refine_prediction(
                refiner,
                points,
                emb,
                obj_idx,
                my_r,
                my_t,
                args.iterations,
                args.num_points,
                device,
            )

            model_points_np = model_points_tensor[0].detach().cpu().numpy()
            target_np = target_tensor[0].detach().cpu().numpy()
            distance = compute_add_distance(model_points_np, target_np, my_r, my_t, obj_id)
            threshold = thresholds.get(obj_id, 0.0)

            is_failure = threshold == 0.0 or distance > threshold
            if is_failure:
                case = render_failure_case(dataset, index, my_r, my_t, distance, threshold)
                if case is not None:
                    candidates.append(case)

    if not candidates:
        return [], {"dataset_size": len(dataset), "evaluated": len(eval_indices)}

    grouped = defaultdict(list)
    for case in candidates:
        grouped[case["object_id"]].append(case)

    selected = []
    selected_indices = set()
    for obj_id in PREFERRED_FAILURE_OBJECTS:
        if grouped[obj_id]:
            grouped[obj_id].sort(key=lambda item: item["failure_ratio"], reverse=True)
            case = grouped[obj_id][0]
            selected.append(case)
            selected_indices.add(case["index"])

    remaining = sorted(
        candidates,
        key=lambda item: (
            item["object_id"] in PREFERRED_FAILURE_OBJECTS,
            item["failure_ratio"],
            item["distance"],
        ),
        reverse=True,
    )
    for case in remaining:
        if len(selected) >= args.num_cases:
            break
        if case["index"] not in selected_indices:
            selected.append(case)
            selected_indices.add(case["index"])

    selected = selected[: args.num_cases]
    compose_failure_grid(selected, output_path)
    metadata = {
        "source": "checkpoint inference",
        "model": model_path,
        "refine_model": refine_model_path,
        "dataset_root": args.dataset_root,
        "device": str(device),
        "evaluated_samples": len(eval_indices),
        "failure_log": args.failure_log if os.path.exists(args.failure_log) else "",
        "known_failures_from_log": len(known_failures),
        "selected_cases": [
            {
                "index": case["index"],
                "object_id": case["object_id"],
                "object_name": case["object_name"],
                "reason": case["reason"],
                "distance": case["distance"],
                "threshold": case["threshold"],
            }
            for case in selected
        ],
    }
    return selected, metadata


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    output_path = os.path.join(args.output_dir, args.output_name)

    existing_cases, checked_sources = find_existing_visualization_cases(args)
    if existing_cases:
        if tile_existing_cases(existing_cases, output_path, args.num_cases):
            metadata = {
                "source": "existing visualization output",
                "selected_images": [case["image_path"] for case in existing_cases[: args.num_cases]],
            }
            _, json_path = status_paths(args.output_dir)
            with open(json_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print("Wrote {0}".format(output_path))
            return

    model_path, refine_model_path = auto_find_checkpoints(args)
    missing = []
    if not model_path:
        missing.append("trained PoseNet checkpoint, e.g. trained_checkpoints/linemod/pose_model_*.pth")
    if not refine_model_path:
        missing.append(
            "trained PoseRefineNet checkpoint, e.g. trained_checkpoints/linemod/pose_refine_model_*.pth"
        )
    if not dataset_available(args.dataset_root):
        missing.append(
            "LINEMOD/Occlusion-LINEMOD dataset root with data/ and models/: {0}".format(
                args.dataset_root
            )
        )

    if missing:
        write_status(
            args.output_dir,
            "need trained checkpoint or existing prediction/visualization results",
            missing=missing,
            sources=checked_sources,
        )
        return

    try:
        selected, metadata = run_linemod_inference(args, model_path, refine_model_path, output_path)
        if not selected:
            write_status(
                args.output_dir,
                "checkpoint inference completed but no failed cases were found in the evaluated subset",
                sources=checked_sources + [args.dataset_root, model_path, refine_model_path],
            )
            return
        _, json_path = status_paths(args.output_dir)
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print("Wrote {0}".format(output_path))
    except Exception as exc:
        write_status(
            args.output_dir,
            "failed to generate failure cases: {0}".format(exc),
            sources=checked_sources + [args.dataset_root, model_path, refine_model_path],
        )


if __name__ == "__main__":
    main()
