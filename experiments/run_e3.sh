#!/bin/bash
# E3: Point Transformer, FiLM global, 1 seed first (probe), then remaining if good, GPU 1
set -e
cd /data1/user/yikun/project/PT-HICNET
PYTHON=/data1/user/yikun/.conda/envs/dl/bin/python
for seed in 42 3407 2026; do
  echo "=== E3 PT film=global seed=${seed} ==="
  CUDA_VISIBLE_DEVICES=1 ${PYTHON} -u scripts/train_pt_hicnet.py \
    --config configs/default.yaml \
    --film_mode global \
    --seed ${seed} \
    --use_wandb
  echo "=== E3 seed=${seed} done ==="
done
echo "E3: all 3 seeds complete"
