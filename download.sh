#!/usr/bin/env bash
set -e

cat <<'MSG'
RAFusion public package

Datasets and trained checkpoints are not bundled with this repository.
Please prepare the required external files manually:

1. LINEMOD:
   datasets/linemod/Linemod_preprocessed/

2. Occlusion LINEMOD / LM-O:
   datasets/bop/lmo/

3. Checkpoints:
   Place RAFusion or DenseFusion checkpoint files in a local ignored folder,
   for example:
   weights/

This script intentionally does not download the original DenseFusion
checkpoints, because they are not RAFusion checkpoints.
MSG
