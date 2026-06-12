#!/bin/bash
# E3-ES-v2: PT + FiLM global + EarlyStopping (patience=20, restore_best), 3 seeds
# seed 42 + 3407: GPU 0 (serial), seed 2026: GPU 1 (parallel when available)
set -e
cd /data1/user/yikun/project/PT-HICNET
PYTHON=/data1/user/yikun/.conda/envs/dl/bin/python

echo "=== E3-ES-v2 seed=42 GPU=0 ==="
CUDA_VISIBLE_DEVICES=0 ${PYTHON} -u scripts/train_pt_hicnet.py \
  --config configs/default.yaml \
  --film_mode global \
  --patience 20 \
  --restore_best \
  --exp_name pt_hicnet_es_v2 \
  --seed 42 \
  --use_wandb
echo "=== E3-ES-v2 seed=42 done ==="

echo "=== E3-ES-v2 seed=3407 GPU=0 ==="
CUDA_VISIBLE_DEVICES=0 ${PYTHON} -u scripts/train_pt_hicnet.py \
  --config configs/default.yaml \
  --film_mode global \
  --patience 20 \
  --restore_best \
  --exp_name pt_hicnet_es_v2 \
  --seed 3407 \
  --use_wandb
echo "=== E3-ES-v2 seed=3407 done ==="

echo "=== E3-ES-v2 seed=2026 GPU=1 ==="
CUDA_VISIBLE_DEVICES=1 ${PYTHON} -u scripts/train_pt_hicnet.py \
  --config configs/default.yaml \
  --film_mode global \
  --patience 20 \
  --restore_best \
  --exp_name pt_hicnet_es_v2 \
  --seed 2026 \
  --use_wandb
echo "=== E3-ES-v2 seed=2026 done ==="

echo "E3-ES-v2: all 3 seeds complete"
