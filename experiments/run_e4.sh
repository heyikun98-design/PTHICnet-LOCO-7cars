#!/bin/bash
# E4: Point Transformer, FiLM deep, 3 seeds, GPU 1
set -e
cd /data1/user/yikun/project/PT-HICNET
PYTHON=/data1/user/yikun/.conda/envs/dl/bin/python
for seed in 42 3407 2026; do
  echo "=== E4 PT film=deep seed=${seed} ==="
  CUDA_VISIBLE_DEVICES=1 ${PYTHON} -u scripts/train_pt_hicnet.py \
    --config configs/default.yaml \
    --film_mode deep \
    --seed ${seed} \
    --use_wandb
  echo "=== E4 seed=${seed} done ==="
done
echo "E4: all 3 seeds complete"
