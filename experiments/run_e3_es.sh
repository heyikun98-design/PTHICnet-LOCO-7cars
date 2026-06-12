#!/bin/bash
# E3-ES: PT + FiLM global + Early Stopping (patience=50), 3 seeds, GPU 0
set -e
cd /data1/user/yikun/project/PT-HICNET
PYTHON=/data1/user/yikun/.conda/envs/dl/bin/python
for seed in 42 3407 2026; do
  echo "=== E3-ES PT film=global patience=50 seed=${seed} ==="
  CUDA_VISIBLE_DEVICES=0 ${PYTHON} -u scripts/train_pt_hicnet.py \
    --config configs/default.yaml \
    --film_mode global \
    --patience 50 \
    --exp_name pt_hicnet_es \
    --seed ${seed} \
    --use_wandb
  echo "=== E3-ES seed=${seed} done ==="
done
echo "E3-ES: all 3 seeds complete"
