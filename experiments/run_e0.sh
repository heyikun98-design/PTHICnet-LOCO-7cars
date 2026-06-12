#!/bin/bash
# E0: PointNet++ baseline, 3 seeds, GPU 0
set -e
cd /data1/user/yikun/project/PT-HICNET
PYTHON=/data1/user/yikun/.conda/envs/dl/bin/python
for seed in 42 3407 2026; do
  echo "=== E0 baseline seed=${seed} ==="
  CUDA_VISIBLE_DEVICES=0 ${PYTHON} -u feather/train_reg_att_props_X70_feather.py \
    --config configs/default.yaml \
    --ablation_mode baseline \
    --seed ${seed} \
    --use_wandb
  echo "=== E0 seed=${seed} done ==="
done
echo "E0: all 3 seeds complete"
