#!/usr/bin/env python3
"""Profile DenseFusion/RAFusion model variants without training.

The script reports trainable parameters, FLOPs, and optional CUDA inference
latency for four architecture variants:

1. DenseFusion baseline
2. DenseFusion + SE
3. DenseFusion + RealFormer
4. RAFusion (SE + RealFormer)

FLOPs are computed with dummy inputs and do not require checkpoints. Timing
uses a checkpoint when one is supplied or auto-detected; otherwise it uses the
randomly initialized architecture and records that fact in the CSV.
"""

from __future__ import print_function

import argparse
import csv
import glob
import json
import os
import sys
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from lib.extractors import resnet18  # noqa: E402
from lib.pspnet import PSPModule, PSPUpsample  # noqa: E402
from lib.realformer import RealFormerEncoder, ResidualMultiHeadAttention  # noqa: E402
from lib.senet import SELayer, se_resnet18  # noqa: E402


VARIANTS = OrderedDict(
    [
        ("DenseFusion baseline", {"use_se": False, "use_realformer": False}),
        ("DenseFusion + SE", {"use_se": True, "use_realformer": False}),
        ("DenseFusion + RealFormer", {"use_se": False, "use_realformer": True}),
        ("RAFusion (SE + RealFormer)", {"use_se": True, "use_realformer": True}),
    ]
)


class ProfilePSPNet(nn.Module):
    """PSPNet image branch with a switchable ResNet18/SE-ResNet18 encoder."""

    def __init__(self, use_se=False, sizes=(1, 2, 3, 6)):
        super(ProfilePSPNet, self).__init__()
        self.feats = se_resnet18() if use_se else resnet18()
        self.psp = PSPModule(512, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)
        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)
        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        features, _ = self.feats(x)
        x = self.psp(features)
        x = self.drop_1(x)
        x = self.up_1(x)
        x = self.drop_2(x)
        x = self.up_2(x)
        x = self.drop_2(x)
        x = self.up_3(x)
        return self.final(x)


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model=1024, max_len=501):
        super(TemporalEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, modal_feats):
        batch_size, seq_len, _ = modal_feats.shape
        indices = torch.arange(seq_len, dtype=torch.long, device=modal_feats.device)
        emb = self.embedding(indices).unsqueeze(0)
        return emb.expand(batch_size, -1, -1)


class PoseNetFeatVariant(nn.Module):
    """DenseFusion point/RGB fusion block with optional RealFormer branch.

    The RealFormer path mirrors the current repository implementation shape
    convention so the profile reflects this codebase rather than a rewritten
    model.
    """

    def __init__(self, num_points, use_realformer=False):
        super(PoseNetFeatVariant, self).__init__()
        self.num_points = num_points
        self.use_realformer = use_realformer

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.e_conv1 = nn.Conv1d(32, 64, 1)
        self.e_conv2 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(256, 512, 1)
        self.conv6 = nn.Conv1d(512, 1024, 1)
        self.ap1 = nn.AvgPool1d(num_points)

        if use_realformer:
            self.temp_emb = TemporalEmbedding(1024, max_len=num_points + 1)
            self.transformer_encoder = RealFormerEncoder(
                d_model=1024,
                num_heads=8,
                expansion_factor=2,
                dropout=0.1,
                num_layers=3,
            )
            self.norm = nn.LayerNorm(1024)
            self.dp = nn.Dropout(0.1)
            self.fc1 = nn.Linear(num_points + 1, num_points)

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv5(pointfeat_2))
        x = F.relu(self.conv6(x))
        pooled = self.ap1(x)

        if self.use_realformer:
            init_global = torch.cat((x, pooled), dim=2).transpose(1, 2)
            mm_src = self.temp_emb(init_global) + init_global
            mm_src = self.dp(self.norm(mm_src))

            # Keep the current repository's RealFormer tensor convention.
            mm_src = mm_src.transpose(0, 1)
            memory = self.transformer_encoder(mm_src)
            memory = memory.transpose(0, 1).transpose(1, 2)
            pooled = self.fc1(memory)
        else:
            pooled = pooled.repeat(1, 1, self.num_points)

        return torch.cat([pointfeat_1, pointfeat_2, pooled], 1)


class PoseNetVariant(nn.Module):
    def __init__(self, num_points=500, num_obj=13, use_se=False, use_realformer=False):
        super(PoseNetVariant, self).__init__()
        self.num_points = num_points
        self.num_obj = num_obj
        self.cnn = ProfilePSPNet(use_se=use_se)
        self.feat = PoseNetFeatVariant(num_points, use_realformer=use_realformer)

        self.conv1_r = nn.Conv1d(1408, 640, 1)
        self.conv1_t = nn.Conv1d(1408, 640, 1)
        self.conv1_c = nn.Conv1d(1408, 640, 1)
        self.conv2_r = nn.Conv1d(640, 256, 1)
        self.conv2_t = nn.Conv1d(640, 256, 1)
        self.conv2_c = nn.Conv1d(640, 256, 1)
        self.conv3_r = nn.Conv1d(256, 128, 1)
        self.conv3_t = nn.Conv1d(256, 128, 1)
        self.conv3_c = nn.Conv1d(256, 128, 1)
        self.conv4_r = nn.Conv1d(128, num_obj * 4, 1)
        self.conv4_t = nn.Conv1d(128, num_obj * 3, 1)
        self.conv4_c = nn.Conv1d(128, num_obj * 1, 1)

    def forward(self, img, points, choose, obj):
        out_img = self.cnn(img)
        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()

        points = points.transpose(2, 1).contiguous()
        fused = self.feat(points, emb)

        rx = F.relu(self.conv1_r(fused))
        tx = F.relu(self.conv1_t(fused))
        cx = F.relu(self.conv1_c(fused))
        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))
        cx = F.relu(self.conv2_c(cx))
        rx = F.relu(self.conv3_r(rx))
        tx = F.relu(self.conv3_t(tx))
        cx = F.relu(self.conv3_c(cx))

        rx = self.conv4_r(rx).view(bs, self.num_obj, 4, self.num_points)
        tx = self.conv4_t(tx).view(bs, self.num_obj, 3, self.num_points)
        cx = torch.sigmoid(self.conv4_c(cx)).view(bs, self.num_obj, 1, self.num_points)

        batch_id = 0
        out_rx = torch.index_select(rx[batch_id], 0, obj[batch_id])
        out_tx = torch.index_select(tx[batch_id], 0, obj[batch_id])
        out_cx = torch.index_select(cx[batch_id], 0, obj[batch_id])

        out_rx = out_rx.contiguous().transpose(2, 1).contiguous()
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()
        out_cx = out_cx.contiguous().transpose(2, 1).contiguous()
        return out_rx, out_tx, out_cx, emb.detach()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile DenseFusion/RAFusion model variants without training."
    )
    parser.add_argument("--output", default="revision_outputs/efficiency_results.csv")
    parser.add_argument("--metadata-output", default="revision_outputs/profile_metadata.json")
    parser.add_argument("--checkpoint", default="", help="Optional PoseNet checkpoint for timing.")
    parser.add_argument(
        "--checkpoint-dir",
        default="",
        help="Optional directory searched recursively for pose_model*.pth.",
    )
    parser.add_argument("--num-points", type=int, default=500)
    parser.add_argument("--num-objects", type=int, default=13)
    parser.add_argument("--image-height", type=int, default=160)
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Timing device. CPU timing is reported only when explicitly requested.",
    )
    parser.add_argument(
        "--allow-cpu-timing",
        action="store_true",
        help="Measure CPU latency when CUDA is unavailable. Disabled by default for paper reporting.",
    )
    parser.add_argument("--threads", type=int, default=0, help="Optional torch CPU thread count.")
    return parser.parse_args()


def build_model(variant, num_points, num_objects):
    cfg = VARIANTS[variant]
    return PoseNetVariant(
        num_points=num_points,
        num_obj=num_objects,
        use_se=cfg["use_se"],
        use_realformer=cfg["use_realformer"],
    )


def make_dummy_inputs(batch_size, num_points, num_objects, image_height, image_width, device):
    img = torch.randn(batch_size, 3, image_height, image_width, device=device)
    points = torch.randn(batch_size, num_points, 3, device=device)
    high = image_height * image_width
    choose = torch.randint(0, high, (batch_size, 1, num_points), dtype=torch.long, device=device)
    obj = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
    obj.clamp_(0, num_objects - 1)
    return img, points, choose, obj


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _conv_flops(module, inputs, output):
    x = inputs[0]
    out = output
    kernel_ops = 1
    for k in module.kernel_size:
        kernel_ops *= k
    kernel_ops *= module.in_channels // module.groups
    return out.numel() * kernel_ops


def _linear_flops(module, inputs, output):
    return output.numel() * module.in_features


def _pool_flops(module, inputs, output):
    x = inputs[0]
    if isinstance(module, nn.AvgPool1d):
        kernel = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
        return output.numel() * kernel
    return x.numel()


def _upsample_flops(module, inputs, output):
    return output.numel() * 4


def _se_scale_flops(module, inputs, output):
    return output.numel()


def _realformer_attention_flops(module, inputs, output):
    x = inputs[0]
    if x.dim() != 3:
        return 0
    batch_size, seq_len, d_model = x.shape
    d_head = d_model // module.num_heads
    qk = batch_size * module.num_heads * seq_len * seq_len * d_head
    av = batch_size * module.num_heads * seq_len * seq_len * d_head
    softmax = batch_size * module.num_heads * seq_len * seq_len
    return qk + av + softmax


def count_flops(model, dummy_inputs):
    flops = {"total": 0}
    handles = []

    def add_hooks(module):
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            handles.append(
                module.register_forward_hook(
                    lambda m, inp, out: flops.__setitem__(
                        "total", flops["total"] + _conv_flops(m, inp, out)
                    )
                )
            )
        elif isinstance(module, nn.Linear):
            handles.append(
                module.register_forward_hook(
                    lambda m, inp, out: flops.__setitem__(
                        "total", flops["total"] + _linear_flops(m, inp, out)
                    )
                )
            )
        elif isinstance(module, (nn.AvgPool1d, nn.AdaptiveAvgPool2d, nn.MaxPool2d)):
            handles.append(
                module.register_forward_hook(
                    lambda m, inp, out: flops.__setitem__(
                        "total", flops["total"] + _pool_flops(m, inp, out)
                    )
                )
            )
        elif isinstance(module, nn.Upsample):
            handles.append(
                module.register_forward_hook(
                    lambda m, inp, out: flops.__setitem__(
                        "total", flops["total"] + _upsample_flops(m, inp, out)
                    )
                )
            )
        elif isinstance(module, SELayer):
            handles.append(
                module.register_forward_hook(
                    lambda m, inp, out: flops.__setitem__(
                        "total", flops["total"] + _se_scale_flops(m, inp, out)
                    )
                )
            )
        elif isinstance(module, ResidualMultiHeadAttention):
            handles.append(
                module.register_forward_hook(
                    lambda m, inp, out: flops.__setitem__(
                        "total", flops["total"] + _realformer_attention_flops(m, inp, out)
                    )
                )
            )

    model.apply(add_hooks)
    model.eval()
    with torch.no_grad():
        model(*dummy_inputs)

    for handle in handles:
        handle.remove()
    return flops["total"]


def normalize_state_dict(state):
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        return {}

    normalized = OrderedDict()
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        normalized[key] = value
    return normalized


def load_matching_checkpoint(model, checkpoint_path, device):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return False, 0, "checkpoint not found"

    try:
        raw_state = torch.load(checkpoint_path, map_location=device)
        state = normalize_state_dict(raw_state)
        current = model.state_dict()
        matched = OrderedDict(
            (key, value)
            for key, value in state.items()
            if key in current and tuple(current[key].shape) == tuple(value.shape)
        )
        if not matched:
            return False, 0, "checkpoint found but no matching tensors"
        current.update(matched)
        model.load_state_dict(current)
        return True, len(matched), "loaded matching tensors"
    except Exception as exc:
        return False, 0, "checkpoint load failed: {0}".format(exc)


def find_checkpoint(args):
    candidates = []
    if args.checkpoint:
        candidates.append(args.checkpoint)
    search_roots = []
    if args.checkpoint_dir:
        search_roots.append(args.checkpoint_dir)
    search_roots.extend(["trained_checkpoints", "trained_models"])

    for root in search_roots:
        if not root:
            continue
        pattern = os.path.join(root, "**", "pose_model*.pth")
        candidates.extend(glob.glob(pattern, recursive=True))

    existing = [path for path in candidates if os.path.exists(path)]
    if not existing:
        return ""
    return sorted(existing)[0]


def select_timing_device(args):
    cuda_available = torch.cuda.is_available()
    if args.device == "cuda":
        return torch.device("cuda") if cuda_available else None
    if args.device == "cpu":
        return torch.device("cpu") if args.allow_cpu_timing else None
    if cuda_available:
        return torch.device("cuda")
    return torch.device("cpu") if args.allow_cpu_timing else None


def measure_latency(model, dummy_inputs, device, warmup, iters):
    model.to(device)
    model.eval()
    dummy_inputs = tuple(x.to(device) for x in dummy_inputs)

    with torch.no_grad():
        for _ in range(warmup):
            model(*dummy_inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            model(*dummy_inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    ms = elapsed * 1000.0 / float(iters)
    fps = 1000.0 / ms if ms > 0 else 0.0
    return ms, fps


def main():
    args = parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    output_path = os.path.abspath(args.output)
    metadata_path = os.path.abspath(args.metadata_output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    checkpoint_path = find_checkpoint(args)
    timing_device = select_timing_device(args)
    flops_device = torch.device("cpu")
    dummy_cpu = make_dummy_inputs(
        1,
        args.num_points,
        args.num_objects,
        args.image_height,
        args.image_width,
        flops_device,
    )

    rows = []
    for variant in VARIANTS.keys():
        model = build_model(variant, args.num_points, args.num_objects)
        params = count_trainable_params(model)
        flops = count_flops(model, dummy_cpu)

        checkpoint_loaded = False
        matched_tensors = 0
        load_message = "no checkpoint supplied or auto-detected"

        if checkpoint_path:
            checkpoint_loaded, matched_tensors, load_message = load_matching_checkpoint(
                model, checkpoint_path, torch.device("cpu")
            )

        latency_ms = ""
        fps = ""
        timing_device_name = "not_measured_no_cuda"
        notes = []

        if checkpoint_loaded:
            weights_source = checkpoint_path
        else:
            weights_source = "random_init"
            notes.append(
                "measured with randomly initialized weights, for computational profiling only"
            )

        if timing_device is not None:
            dummy_timing = make_dummy_inputs(
                1,
                args.num_points,
                args.num_objects,
                args.image_height,
                args.image_width,
                timing_device,
            )
            latency_ms, fps = measure_latency(
                model, dummy_timing, timing_device, args.warmup, args.iters
            )
            timing_device_name = str(timing_device)
        else:
            notes.append(
                "CUDA is unavailable; inference time should be measured on the same GPU environment as the paper"
            )

        if load_message:
            notes.append(load_message)

        rows.append(
            {
                "model": variant,
                "trainable_params_m": "{0:.3f}".format(params / 1e6),
                "flops_g": "{0:.3f}".format(flops / 1e9),
                "inference_time_ms": "" if latency_ms == "" else "{0:.3f}".format(latency_ms),
                "fps": "" if fps == "" else "{0:.3f}".format(fps),
                "timing_device": timing_device_name,
                "weights_source": weights_source,
                "checkpoint_loaded": str(bool(checkpoint_loaded)),
                "matched_checkpoint_tensors": str(matched_tensors),
                "input_resolution": "{0}x{1}".format(args.image_height, args.image_width),
                "num_points": str(args.num_points),
                "num_objects": str(args.num_objects),
                "batch_size": "1",
                "warmup_iterations": str(args.warmup),
                "timed_iterations": str(args.iters),
                "notes": "; ".join(notes),
            }
        )

    fieldnames = [
        "model",
        "trainable_params_m",
        "flops_g",
        "inference_time_ms",
        "fps",
        "timing_device",
        "weights_source",
        "checkpoint_loaded",
        "matched_checkpoint_tensors",
        "input_resolution",
        "num_points",
        "num_objects",
        "batch_size",
        "warmup_iterations",
        "timed_iterations",
        "notes",
    ]

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "output": output_path,
        "checkpoint_auto_detected": checkpoint_path,
        "cuda_available": torch.cuda.is_available(),
        "timing_device": str(timing_device) if timing_device is not None else "not measured",
        "flops_note": "FLOPs are estimated from one dummy PoseNet forward pass; convolution and linear multiply-adds are counted in profiler-style MAC units.",
        "profile_scope": "PoseNet estimator only; data loading and iterative refinement are excluded.",
        "dummy_input": {
            "batch_size": 1,
            "num_points": args.num_points,
            "num_objects": args.num_objects,
            "image_height": args.image_height,
            "image_width": args.image_width,
        },
    }
    with open(metadata_path, "w") as meta_file:
        json.dump(metadata, meta_file, indent=2)

    print("Wrote {0}".format(output_path))
    print("Wrote {0}".format(metadata_path))


if __name__ == "__main__":
    main()
