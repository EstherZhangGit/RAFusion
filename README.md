# RAFusion

This repository contains the source-code release for **RAFusion**, an RGB-D 6D object pose estimation model built on the DenseFusion pipeline. RAFusion keeps the DenseFusion-style dense RGB-D fusion and iterative pose refinement framework, and adds residual-attention components for stronger global context modeling.

This public package provides the model, data loaders, training entry point, and evaluation utilities. Large datasets, trained checkpoints, local experiment logs, and private revision files are not included. To reproduce reported results, prepare the required benchmark datasets and checkpoint files separately.

## Main Components

- `lib/network.py`: RAFusion pose estimation network.
- `lib/senet.py`: SE-ResNet blocks used in the RGB feature branch.
- `lib/realformer.py`: RealFormer global-context attention module.
- `lib/network_densefusion.py`: DenseFusion baseline implementation used for comparison.
- `lib/network_vovnet.py` and `lib/vovnet.py`: VoVNet-based experimental variant.
- `tools/profile_model.py`: computational efficiency profiling for DenseFusion, DenseFusion+SE, DenseFusion+RealFormer, and RAFusion.
- `tools/visualize_failure_cases.py`: failure-case visualization utility for revision experiments.
- `tools/eval_linemod.py`: LINEMOD evaluation script.
- `tools/eval_occlusion_linemod.py`: Occlusion LINEMOD / LM-O evaluation script.
- `tools/run_inference_stability.py`: multi-seed inference-level stability evaluation. This requires explicit checkpoint paths.
- `tools/train.py`: RAFusion training entry point using the DenseFusion-style two-stage pose estimation and refinement pipeline.

## What Is Not Included

The following files are intentionally excluded from this GitHub-ready package:

- LINEMOD, Occlusion LINEMOD / LM-O, YCB-Video, or other full datasets.
- Trained `.pth` checkpoint files.
- Local `weights/`, `trained_checkpoints/`, `Linemod_preprocessed/`, BOP dataset folders, and generated predictions.
- Manuscript revision drafts, response letters, reviewer files, and private experiment archives.

## Expected Data Layout

Place external datasets outside the GitHub repository or under ignored local paths. For example:

```text
datasets/
  linemod/
    Linemod_preprocessed/
  bop/
    lmo/
weights/
  linemod/
    rafusion_pose_model.pth
    rafusion_pose_refine_model.pth
```

If you use a different local path, update the dataset root and checkpoint arguments in the corresponding script. Some legacy scripts still use `trained_checkpoints/` or `trained_models/` as default search paths; pass `--model` and `--refine_model` explicitly when evaluating.

## Basic Usage

Profile model efficiency without retraining:

```bash
python tools/profile_model.py
```

Evaluate on LINEMOD using an existing checkpoint:

```bash
python tools/eval_linemod.py \
  --dataset_root datasets/linemod/Linemod_preprocessed \
  --model weights/linemod/rafusion_pose_model.pth \
  --refine_model weights/linemod/rafusion_pose_refine_model.pth
```

Evaluate on Occlusion LINEMOD / LM-O using existing data and checkpoint:

```bash
python tools/eval_occlusion_linemod.py \
  --dataset_root datasets/bop/lmo \
  --model weights/linemod/rafusion_pose_model.pth \
  --refine_model weights/linemod/rafusion_pose_refine_model.pth
```

Run inference-level random-seed stability evaluation:

```bash
python tools/run_inference_stability.py \
  --dataset_root datasets/bop/lmo \
  --model weights/linemod/rafusion_pose_model.pth \
  --refine_model weights/linemod/rafusion_pose_refine_model.pth
```

Generate failure-case visualization from existing predictions/checkpoints:

```bash
python tools/visualize_failure_cases.py --help
```

## Notes For Reproducibility

- Efficiency profiling can compute parameter counts and FLOPs with dummy inputs and does not require retraining.
- Evaluation and visualization scripts require externally prepared datasets and checkpoint files; this repository does not include ready-to-run weights.
- Inference time should be measured on the same GPU environment used for the reported experiment when comparing speed.
- Failure-case visualization requires either trained checkpoints or existing prediction/evaluation result files.
- LINEMOD and Occlusion LINEMOD use fixed benchmark splits in common evaluation protocols; multi-seed inference stability is included as a practical alternative to k-fold cross-validation for the revision experiments.

## Acknowledgement

This project is based on the DenseFusion code structure and evaluation pipeline:

```bibtex
@inproceedings{wang2019densefusion,
  title={DenseFusion: 6D Object Pose Estimation by Iterative Dense Fusion},
  author={Wang, Chen and Xu, Danfei and Zhu, Yuke and Martin-Martin, Roberto and Lu, Cewu and Fei-Fei, Li and Savarese, Silvio},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

Please also cite the RAFusion paper when using this repository for RAFusion-related results.

## License

This repository follows the original project license in `LICENSE`.
