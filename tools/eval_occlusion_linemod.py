import _init_paths
import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
from torch.autograd import Variable

from datasets.occlusion_linemod.dataset import LINEMOD_OBJLIST
from datasets.occlusion_linemod.dataset import PoseDataset as PoseDataset_lmo
from lib.transformations import quaternion_from_matrix, quaternion_matrix


OBJECT_NAMES = {
    1: "ape",
    5: "can",
    6: "cat",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
}
SYMMETRIC_OBJ_IDS = {10, 11}


def parse_args():
    parser = argparse.ArgumentParser("Evaluate RAFusion/DenseFusion on BOP-format LM-O.")
    parser.add_argument("--dataset_root", default="datasets/bop/lmo")
    parser.add_argument("--model", required=True, help="PoseNet checkpoint path")
    parser.add_argument("--refine_model", default="", help="PoseRefineNet checkpoint path")
    parser.add_argument(
        "--arch",
        default="rafusion",
        choices=["rafusion", "densefusion", "vovnet"],
        help="Network architecture used by the checkpoint.",
    )
    parser.add_argument("--output_dir", default="revision_outputs")
    parser.add_argument("--num_points", type=int, default=500)
    parser.add_argument("--num_objects", type=int, default=13)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max_samples", type=int, default=-1, help="Limit samples for smoke tests")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for point/model sampling.")
    parser.add_argument(
        "--detector_results",
        default="",
        help=(
            "Optional detector output file. CSV/JSON rows should contain scene_id, "
            "im_id, obj_id, score and bbox_est/bbox or x,y,w,h."
        ),
    )
    parser.add_argument("--detector_score_threshold", type=float, default=0.0)
    parser.add_argument(
        "--detector_match",
        default="score",
        choices=["score", "iou"],
        help="How to choose one detection when multiple boxes match the target class.",
    )
    parser.add_argument(
        "--detector_fallback_to_gt",
        action="store_true",
        help="Use GT visible masks when detector output is missing for a target.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def get_network_classes(arch):
    if arch == "densefusion":
        from lib.network_densefusion import PoseNet, PoseRefineNet
    elif arch == "vovnet":
        from lib.network_vovnet import PoseNet, PoseRefineNet
    else:
        from lib.network import PoseNet, PoseRefineNet
    return PoseNet, PoseRefineNet


def select_device(requested):
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(model, checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    model.load_state_dict(cleaned)


def pose_from_network(estimator, refiner, data, args, device):
    points, choose, img, target, model_points, idx = data
    points = Variable(points).to(device)
    choose = Variable(choose).to(device)
    img = Variable(img).to(device)
    idx = Variable(idx).to(device)

    pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
    pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, args.num_points, 1)
    pred_c = pred_c.view(1, args.num_points)
    _, which_max = torch.max(pred_c, 1)
    pred_t = pred_t.view(args.num_points, 1, 3)

    my_r = pred_r[0][which_max[0]].view(-1).detach().cpu().numpy()
    my_t = (
        points.view(args.num_points, 1, 3) + pred_t
    )[which_max[0]].view(-1).detach().cpu().numpy()

    if refiner is not None and args.iterations > 0:
        for _ in range(args.iterations):
            t_tensor = (
                torch.from_numpy(my_t.astype(np.float32))
                .to(device)
                .view(1, 3)
                .repeat(args.num_points, 1)
                .contiguous()
                .view(1, args.num_points, 3)
            )
            my_mat = quaternion_matrix(my_r)
            r_tensor = (
                torch.from_numpy(my_mat[:3, :3].astype(np.float32))
                .to(device)
                .view(1, 3, 3)
            )
            my_mat[0:3, 3] = my_t
            new_points = torch.bmm((points - t_tensor), r_tensor).contiguous()
            pred_r, pred_t = refiner(new_points, emb, idx)
            pred_r = pred_r.view(1, 1, -1)
            pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, 1, 1)
            delta_r = pred_r.view(-1).detach().cpu().numpy()
            delta_t = pred_t.view(-1).detach().cpu().numpy()

            delta_mat = quaternion_matrix(delta_r)
            delta_mat[0:3, 3] = delta_t
            final_mat = np.dot(my_mat, delta_mat)
            final_r = final_mat.copy()
            final_r[0:3, 3] = 0
            my_r = quaternion_from_matrix(final_r, True)
            my_t = np.array([final_mat[0, 3], final_mat[1, 3], final_mat[2, 3]])

    pred_r_mat = quaternion_matrix(my_r)[:3, :3]
    return pred_r_mat, my_t


def add_distance(model_points, gt_r, gt_t, pred_r, pred_t, symmetric):
    gt = np.dot(model_points, gt_r.T) + gt_t
    pred = np.dot(model_points, pred_r.T) + pred_t
    if symmetric:
        diff = pred[:, None, :] - gt[None, :, :]
        nearest = np.min(np.linalg.norm(diff, axis=2), axis=1)
        return float(np.mean(nearest))
    return float(np.mean(np.linalg.norm(pred - gt, axis=1)))


def project(points, r_mat, t_vec, cam):
    pts = np.dot(points, r_mat.T) + t_vec
    z = np.maximum(pts[:, 2], 1e-6)
    u = pts[:, 0] * cam["fx"] / z + cam["cx"]
    v = pts[:, 1] * cam["fy"] / z + cam["cy"]
    return np.stack([u, v], axis=1)


def reprojection_error(model_points, gt_r, gt_t, pred_r, pred_t, cam):
    gt_2d = project(model_points, gt_r, gt_t, cam)
    pred_2d = project(model_points, pred_r, pred_t, cam)
    return float(np.mean(np.linalg.norm(gt_2d - pred_2d, axis=1)))


def init_result():
    return {
        "total": 0,
        "add_success": 0,
        "reproj_success": 0,
        "add_sum": 0.0,
        "reproj_sum": 0.0,
        "lost": 0,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = select_device(args.device)

    dataset = PoseDataset_lmo(
        "eval",
        args.num_points,
        False,
        args.dataset_root,
        0.0,
        True,
        detector_results=args.detector_results,
        detector_score_threshold=args.detector_score_threshold,
        detector_match=args.detector_match,
        detector_fallback_to_gt=args.detector_fallback_to_gt,
        sample_seed=args.seed,
    )
    end_index = len(dataset) if args.max_samples < 0 else min(len(dataset), args.start_index + args.max_samples)
    indices = list(range(max(0, args.start_index), end_index))
    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    PoseNet, PoseRefineNet = get_network_classes(args.arch)

    estimator = PoseNet(num_points=args.num_points, num_obj=args.num_objects).to(device)
    load_checkpoint(estimator, args.model, device)
    estimator.eval()

    refiner = None
    if args.refine_model:
        refiner = PoseRefineNet(num_points=args.num_points, num_obj=args.num_objects).to(device)
        load_checkpoint(refiner, args.refine_model, device)
        refiner.eval()

    per_object = defaultdict(init_result)
    overall = init_result()
    model_labels = {
        "densefusion": "DenseFusion",
        "rafusion": "RAFusion",
        "vovnet": "VOVNet-Transformer",
    }
    model_label = model_labels[args.arch]
    suffix = "_{0}{1}".format(args.arch, "_smoke" if args.max_samples > 0 else "")
    log_path = os.path.join(args.output_dir, "occlusion_linemod_eval_logs{0}.txt".format(suffix))
    per_sample_path = os.path.join(args.output_dir, "occlusion_linemod_predictions{0}.csv".format(suffix))
    csv_path = os.path.join(args.output_dir, "occlusion_linemod_eval_results{0}.csv".format(suffix))
    metadata_path = os.path.join(args.output_dir, "occlusion_linemod_eval_metadata{0}.json".format(suffix))

    start_time = time.time()
    with open(log_path, "w", encoding="utf-8") as log_f, open(
        per_sample_path, "w", newline="", encoding="utf-8"
    ) as sample_f:
        sample_writer = csv.DictWriter(
            sample_f,
            fieldnames=[
                "dataset_index",
                "scene_id",
                "im_id",
                "gt_id",
                "object_id",
                "object_name",
                "add_s",
                "add_threshold",
                "add_success",
                "reprojection_2d",
                "reprojection_success",
                "input_source",
                "detector_score",
                "detector_bbox",
            ],
        )
        sample_writer.writeheader()

        with torch.no_grad():
            for local_i, data in enumerate(dataloader):
                dataset_index = indices[local_i]
                meta = dataset.get_sample_meta(dataset_index)
                obj_id = int(meta["obj_id"])
                object_name = OBJECT_NAMES.get(obj_id, "object_{0}".format(obj_id))
                input_source, detection = dataset.get_input_source(dataset_index)

                if len(data[0].size()) == 2:
                    per_object[obj_id]["lost"] += 1
                    overall["lost"] += 1
                    if args.detector_results:
                        add_threshold = 0.1 * dataset.get_diameter(obj_id)
                        for result in (per_object[obj_id], overall):
                            result["total"] += 1
                        sample_writer.writerow(
                            {
                                "dataset_index": dataset_index,
                                "scene_id": meta["scene_id"],
                                "im_id": meta["im_id"],
                                "gt_id": meta["gt_id"],
                                "object_id": obj_id,
                                "object_name": object_name,
                                "add_s": "",
                                "add_threshold": add_threshold,
                                "add_success": 0,
                                "reprojection_2d": "",
                                "reprojection_success": 0,
                                "input_source": input_source,
                                "detector_score": "" if detection is None else detection.get("score", ""),
                                "detector_bbox": "" if detection is None else detection.get("bbox", ""),
                            }
                        )
                        log_f.write(
                            "No.{0} {1} NOT Pass! Lost detection/depth from {2}\n".format(
                                dataset_index, object_name, input_source
                            )
                        )
                        log_f.flush()
                    continue

                pred_r, pred_t = pose_from_network(estimator, refiner, data, args, device)
                gt_r = np.resize(np.array(meta["gt"]["cam_R_m2c"], dtype=np.float32), (3, 3))
                gt_t = np.array(meta["gt"]["cam_t_m2c"], dtype=np.float32) / 1000.0
                model_points = data[4][0].detach().cpu().numpy()
                cam = dataset._camera(meta)

                add_s = add_distance(
                    model_points,
                    gt_r,
                    gt_t,
                    pred_r,
                    pred_t,
                    symmetric=obj_id in SYMMETRIC_OBJ_IDS,
                )
                reproj = reprojection_error(model_points, gt_r, gt_t, pred_r, pred_t, cam)
                add_threshold = 0.1 * dataset.get_diameter(obj_id)
                add_ok = add_s < add_threshold
                reproj_ok = reproj < 5.0

                for result in (per_object[obj_id], overall):
                    result["total"] += 1
                    result["add_success"] += int(add_ok)
                    result["reproj_success"] += int(reproj_ok)
                    result["add_sum"] += add_s
                    result["reproj_sum"] += reproj

                row = {
                    "dataset_index": dataset_index,
                    "scene_id": meta["scene_id"],
                    "im_id": meta["im_id"],
                    "gt_id": meta["gt_id"],
                    "object_id": obj_id,
                    "object_name": object_name,
                    "add_s": add_s,
                    "add_threshold": add_threshold,
                    "add_success": int(add_ok),
                    "reprojection_2d": reproj,
                    "reprojection_success": int(reproj_ok),
                    "input_source": input_source,
                    "detector_score": "" if detection is None else detection.get("score", ""),
                    "detector_bbox": "" if detection is None else detection.get("bbox", ""),
                }
                sample_writer.writerow(row)
                line = (
                    "No.{0} {1} ADD(-S): {2:.6f}/{3:.6f} {4}, 2D: {5:.3f}px {6}".format(
                        dataset_index,
                        object_name,
                        add_s,
                        add_threshold,
                        "Pass" if add_ok else "NOT Pass",
                        reproj,
                        "Pass" if reproj_ok else "NOT Pass",
                    )
                )
                log_f.write(line + "\n")
                log_f.flush()
                if not args.quiet:
                    print(line)

    rows = []
    for obj_id in sorted(per_object.keys()):
        result = per_object[obj_id]
        total = result["total"]
        rows.append(
            {
                "model": model_label,
                "object_id": obj_id,
                "object_name": OBJECT_NAMES.get(obj_id, "object_{0}".format(obj_id)),
                "add_s": result["add_success"] / total if total else "",
                "reprojection_2d": result["reproj_success"] / total if total else "",
                "mean_add_s_distance": result["add_sum"] / total if total else "",
                "mean_reprojection_2d_px": result["reproj_sum"] / total if total else "",
                "success_count_add_s": result["add_success"],
                "success_count_2d": result["reproj_success"],
                "total_count": total,
                "lost_count": result["lost"],
            }
        )

    total = overall["total"]
    rows.append(
        {
            "model": model_label,
            "object_id": "ALL",
            "object_name": "ALL",
            "add_s": overall["add_success"] / total if total else "",
            "reprojection_2d": overall["reproj_success"] / total if total else "",
            "mean_add_s_distance": overall["add_sum"] / total if total else "",
            "mean_reprojection_2d_px": overall["reproj_sum"] / total if total else "",
            "success_count_add_s": overall["add_success"],
            "success_count_2d": overall["reproj_success"],
            "total_count": total,
            "lost_count": overall["lost"],
        }
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "arch": args.arch,
        "seed": args.seed,
        "dataset_root": args.dataset_root,
        "model": args.model,
        "refine_model": args.refine_model,
        "device": str(device),
        "detector_results": args.detector_results,
        "detector_score_threshold": args.detector_score_threshold,
        "detector_match": args.detector_match,
        "detector_fallback_to_gt": args.detector_fallback_to_gt,
        "num_points": args.num_points,
        "num_objects": args.num_objects,
        "iterations": args.iterations,
        "dataset_size": len(dataset),
        "evaluated_samples": len(indices),
        "start_index": args.start_index,
        "max_samples": args.max_samples,
        "metric_note": "ADD(-S) and 2D reprojection are computed on the sampled 500 model points, matching the local DenseFusion-style eval script.",
        "elapsed_seconds": time.time() - start_time,
        "outputs": {
            "summary_csv": csv_path,
            "per_sample_csv": per_sample_path,
            "log": log_path,
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Wrote {0}".format(csv_path))
    print("Wrote {0}".format(per_sample_path))
    print("Wrote {0}".format(metadata_path))


if __name__ == "__main__":
    main()
