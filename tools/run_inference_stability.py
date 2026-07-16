#!/usr/bin/env python3
"""Run inference-level stability evaluation over multiple random seeds.

The script repeats the same LM-O evaluation protocol with different random
seeds, then reports mean/std for ADD(-S), 2D reprojection, and evaluation time.
This is a lightweight alternative when full multi-seed retraining is not
available.
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import subprocess
import sys
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    parser = argparse.ArgumentParser("Multi-seed inference-level stability runner.")
    parser.add_argument("--dataset_root", default="datasets/bop/lmo")
    parser.add_argument("--model", default="trained_checkpoints/linemod/rafusion_pose_model.pth")
    parser.add_argument("--refine_model", default="trained_checkpoints/linemod/rafusion_pose_refine_model.pth")
    parser.add_argument("--arch", default="rafusion", choices=["rafusion", "densefusion"])
    parser.add_argument("--output_dir", default="revision_outputs/inference_stability_lmo")
    parser.add_argument("--seeds", default="2026,2027,2028")
    parser.add_argument("--device", default="cuda", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--wait_pid", type=int, default=-1, help="Optional process id to wait for before running.")
    return parser.parse_args()


def wait_for_pid(pid):
    if pid <= 0:
        return
    while True:
        result = subprocess.call(["ps", "-p", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result != 0:
            return
        time.sleep(30)


def read_overall(result_csv):
    with open(result_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    overall = [row for row in rows if row["object_id"] == "ALL"][0]
    return {
        "add_s": float(overall["add_s"]),
        "reprojection_2d": float(overall["reprojection_2d"]),
        "mean_add_s_distance": float(overall["mean_add_s_distance"]),
        "mean_reprojection_2d_px": float(overall["mean_reprojection_2d_px"]),
        "total_count": int(overall["total_count"]),
    }


def read_metadata(metadata_json):
    with open(metadata_json, "r", encoding="utf-8") as f:
        return json.load(f)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std(values):
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return (sum((x - m) ** 2 for x in values) / (len(values) - 1)) ** 0.5


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    wait_for_pid(args.wait_pid)

    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    processes = []
    eval_script = os.path.join(ROOT_DIR, "tools", "eval_occlusion_linemod.py")

    for seed in seeds:
        seed_dir = os.path.join(args.output_dir, "seed_{0}".format(seed))
        os.makedirs(seed_dir, exist_ok=True)
        cmd = [
            sys.executable,
            eval_script,
            "--arch",
            args.arch,
            "--dataset_root",
            args.dataset_root,
            "--model",
            args.model,
            "--output_dir",
            seed_dir,
            "--device",
            args.device,
            "--seed",
            str(seed),
            "--quiet",
        ]
        if args.refine_model:
            cmd.extend(["--refine_model", args.refine_model])
        if args.max_samples > 0:
            cmd.extend(["--max_samples", str(args.max_samples)])

        if args.parallel:
            processes.append((seed, seed_dir, subprocess.Popen(cmd, cwd=ROOT_DIR)))
        else:
            subprocess.check_call(cmd, cwd=ROOT_DIR)
            processes.append((seed, seed_dir, None))

    for seed, _, process in processes:
        if process is not None:
            ret = process.wait()
            if ret != 0:
                raise RuntimeError("Seed {0} failed with exit code {1}".format(seed, ret))

    suffix = "_{0}{1}".format(args.arch, "_smoke" if args.max_samples > 0 else "")
    rows = []
    for seed in seeds:
        seed_dir = os.path.join(args.output_dir, "seed_{0}".format(seed))
        result_csv = os.path.join(seed_dir, "occlusion_linemod_eval_results{0}.csv".format(suffix))
        metadata_json = os.path.join(seed_dir, "occlusion_linemod_eval_metadata{0}.json".format(suffix))
        result = read_overall(result_csv)
        metadata = read_metadata(metadata_json)
        elapsed = float(metadata["elapsed_seconds"])
        total = int(metadata["evaluated_samples"])
        rows.append(
            {
                "seed": seed,
                "add_s": result["add_s"],
                "reprojection_2d": result["reprojection_2d"],
                "mean_add_s_distance": result["mean_add_s_distance"],
                "mean_reprojection_2d_px": result["mean_reprojection_2d_px"],
                "evaluated_samples": total,
                "elapsed_seconds": elapsed,
                "eval_ms_per_sample": elapsed * 1000.0 / total if total else "",
                "output_dir": seed_dir,
            }
        )

    per_seed_path = os.path.join(args.output_dir, "inference_stability_per_seed.csv")
    with open(per_seed_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metrics = [
        "add_s",
        "reprojection_2d",
        "mean_add_s_distance",
        "mean_reprojection_2d_px",
        "eval_ms_per_sample",
    ]
    summary_rows = []
    for metric in metrics:
        values = [float(row[metric]) for row in rows if row[metric] != ""]
        summary_rows.append(
            {
                "metric": metric,
                "mean": mean(values),
                "std": std(values),
                "num_runs": len(values),
                "seeds": args.seeds,
                "protocol_note": "Same LM-O split and checkpoint; only random point/model sampling seed is changed.",
            }
        )

    summary_path = os.path.join(args.output_dir, "inference_stability_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    readme_path = os.path.join(args.output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# Inference-Level Stability Evaluation\n\n")
        f.write("This folder contains repeated LM-O inference evaluations with different random seeds. ")
        f.write("The benchmark split, checkpoint, evaluation script, and metrics are fixed; only random point/model sampling changes.\n\n")
        f.write("This is an inference-level stability probe, not full multi-seed retraining.\n\n")
        f.write("- Per-seed results: `inference_stability_per_seed.csv`\n")
        f.write("- Mean/std summary: `inference_stability_summary.csv`\n")

    print("Wrote", per_seed_path)
    print("Wrote", summary_path)


if __name__ == "__main__":
    main()
