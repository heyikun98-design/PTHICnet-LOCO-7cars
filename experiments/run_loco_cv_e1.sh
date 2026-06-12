#!/bin/bash
# E1 LOCO-CV: PointNet++ + EarlyFusion
# Usage: bash run_loco_cv_e1.sh [smoke]  — smoke = C201 only
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON=/data1/user/yikun/.conda/envs/dl/bin/python
SCREEN=/data1/user/yikun/.conda/envs/dl/bin/screen
SMOKE="${1:-}"

echo "=== E1 LOCO-CV (early_fusion_clean) ==="

if [ "$SMOKE" = "smoke" ]; then
    echo "[Smoke] C201 only"
    cat > /tmp/loco_e1_smoke.sh << 'INNER'
#!/bin/bash
cd /data1/user/yikun/project/PT-HICNET
PYTHON=/data1/user/yikun/.conda/envs/dl/bin/python
CUDA_VISIBLE_DEVICES=0 ${PYTHON} -u feather/train_reg_att_props_X70_feather.py \
  --config configs/default.yaml --ablation_mode early_fusion_clean \
  --patience 50 --seed 42 \
  --test_vehicles C201 --val_split 0.15 --split_seed 2026 \
  --exp_name pt_hicnet_loco_e1_fold0_C201_seed42
echo "=== E1 Smoke done ==="
INNER
    chmod +x /tmp/loco_e1_smoke.sh
    $SCREEN -S E1-SMOKE -dm bash /tmp/loco_e1_smoke.sh
    sleep 2
    $SCREEN -ls | grep E1
    exit 0
fi

# Full 7-fold
cat > /tmp/loco_e1_gpu0.sh << 'INNER'
#!/bin/bash
cd /data1/user/yikun/project/PT-HICNET
PYTHON=/data1/user/yikun/.conda/envs/dl/bin/python
for pair in "0 C201" "1 EP32" "2 JX65" "3 CY02C"; do
  fold=${pair% *}; vehicle=${pair#* }
  echo "=== E1 Fold ${fold}: ${vehicle} ==="
  CUDA_VISIBLE_DEVICES=0 ${PYTHON} -u feather/train_reg_att_props_X70_feather.py \
    --config configs/default.yaml --ablation_mode early_fusion_clean \
    --patience 50 --seed 42 \
    --test_vehicles "${vehicle}" --val_split 0.15 --split_seed 2026 \
    --exp_name "pt_hicnet_loco_e1_fold${fold}_${vehicle}_seed42"
  echo "=== E1 Fold ${fold} done ==="
done
INNER

cat > /tmp/loco_e1_gpu1.sh << 'INNER'
#!/bin/bash
cd /data1/user/yikun/project/PT-HICNET
PYTHON=/data1/user/yikun/.conda/envs/dl/bin/python
for pair in "4 M6" "5 S50EVK" "6 FX11"; do
  fold=${pair% *}; vehicle=${pair#* }
  echo "=== E1 Fold ${fold}: ${vehicle} ==="
  CUDA_VISIBLE_DEVICES=1 ${PYTHON} -u feather/train_reg_att_props_X70_feather.py \
    --config configs/default.yaml --ablation_mode early_fusion_clean \
    --patience 50 --seed 42 \
    --test_vehicles "${vehicle}" --val_split 0.15 --split_seed 2026 \
    --exp_name "pt_hicnet_loco_e1_fold${fold}_${vehicle}_seed42"
  echo "=== E1 Fold ${fold} done ==="
done
INNER

chmod +x /tmp/loco_e1_gpu0.sh /tmp/loco_e1_gpu1.sh
$SCREEN -S E1-GPU0 -dm bash /tmp/loco_e1_gpu0.sh
$SCREEN -S E1-GPU1 -dm bash /tmp/loco_e1_gpu1.sh
sleep 2
$SCREEN -ls | grep E1
