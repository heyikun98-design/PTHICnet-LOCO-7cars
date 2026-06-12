#!/bin/bash
# E2: Point Transformer, no FiLM, 3 seeds, GPU 1
set -e
cd /data1/user/yikun/project/PT-HICNET
PYTHON=/data1/user/yikun/.conda/envs/dl/bin/python
for seed in 42 3407 2026; do
  echo "=== E2 PT film=none seed=${seed} ==="
  CUDA_VISIBLE_DEVICES=1 ${PYTHON} -u scripts/train_pt_hicnet.py \
    --config configs/default.yaml \
    --film_mode none \
    --seed ${seed} \
    --use_wandb
  echo "=== E2 seed=${seed} done ==="
done
echo "E2: all 3 seeds complete"
