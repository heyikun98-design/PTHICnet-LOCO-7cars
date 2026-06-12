#!/bin/bash
# E1: PointNet++ early_fusion_clean, 3 seeds, GPU 0
set -e
cd /data1/user/yikun/project/PT-HICNET
PYTHON=/data1/user/yikun/.conda/envs/dl/bin/python
for seed in 42 3407 2026; do
  echo "=== E1 early_fusion_clean seed=${seed} ==="
  CUDA_VISIBLE_DEVICES=0 ${PYTHON} -u feather/train_reg_att_props_X70_feather.py \
    --config configs/default.yaml \
    --ablation_mode early_fusion_clean \
    --seed ${seed} \
    --use_wandb
  echo "=== E1 seed=${seed} done ==="
done
echo "E1: all 3 seeds complete"
