#!/bin/bash
# E0 LOCO-CV: PointNet++ baseline (no EarlyFusion)
# Usage: bash run_loco_cv_e0.sh [smoke]  — smoke = C201 only
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON=/data1/user/yikun/.conda/envs/dl/bin/python
SCREEN=/data1/user/yikun/.conda/envs/dl/bin/screen
SMOKE="${1:-}"

echo "=== E0 LOCO-CV (baseline) ==="

if [ "$SMOKE" = "smoke" ]; then
    echo "[Smoke] C201 only"
    cat > /tmp/loco_e0_smoke.sh << 'INNER'
#!/bin/bash
cd /data1/user/yikun/project/PT-HICNET
PYTHON=/data1/user/yikun/.conda/envs/dl/bin/python
CUDA_VISIBLE_DEVICES=0 ${PYTHON} -u feather/train_reg_att_props_X70_feather.py \
  --config configs/default.yaml --ablation_mode baseline \
  --patience 50 --seed 42 \
  --test_vehicles C201 --val_split 0.15 --split_seed 2026 \
  --exp_name pt_hicnet_loco_e0_fold0_C201_seed42
echo "=== E0 Smoke done ==="
INNER
    chmod +x /tmp/loco_e0_smoke.sh
    $SCREEN -S E0-SMOKE -dm bash /tmp/loco_e0_smoke.sh
    sleep 2
    $SCREEN -ls | grep E0
    exit 0
fi

# Full 7-fold
cat > /tmp/loco_e0_gpu0.sh << 'INNER'
#!/bin/bash
cd /data1/user/yikun/project/PT-HICNET
PYTHON=/data1/user/yikun/.conda/envs/dl/bin/python
for pair in "0 C201" "1 EP32" "2 JX65" "3 CY02C"; do
  fold=${pair% *}; vehicle=${pair#* }
  echo "=== E0 Fold ${fold}: ${vehicle} ==="
  CUDA_VISIBLE_DEVICES=0 ${PYTHON} -u feather/train_reg_att_props_X70_feather.py \
    --config configs/default.yaml --ablation_mode baseline \
    --patience 50 --seed 42 \
    --test_vehicles "${vehicle}" --val_split 0.15 --split_seed 2026 \
    --exp_name "pt_hicnet_loco_e0_fold${fold}_${vehicle}_seed42"
  echo "=== E0 Fold ${fold} done ==="
done
INNER

cat > /tmp/loco_e0_gpu1.sh << 'INNER'
#!/bin/bash
cd /data1/user/yikun/project/PT-HICNET
PYTHON=/data1/user/yikun/.conda/envs/dl/bin/python
for pair in "4 M6" "5 S50EVK" "6 FX11"; do
  fold=${pair% *}; vehicle=${pair#* }
  echo "=== E0 Fold ${fold}: ${vehicle} ==="
  CUDA_VISIBLE_DEVICES=1 ${PYTHON} -u feather/train_reg_att_props_X70_feather.py \
    --config configs/default.yaml --ablation_mode baseline \
    --patience 50 --seed 42 \
    --test_vehicles "${vehicle}" --val_split 0.15 --split_seed 2026 \
    --exp_name "pt_hicnet_loco_e0_fold${fold}_${vehicle}_seed42"
  echo "=== E0 Fold ${fold} done ==="
done
INNER

chmod +x /tmp/loco_e0_gpu0.sh /tmp/loco_e0_gpu1.sh
$SCREEN -S E0-GPU0 -dm bash /tmp/loco_e0_gpu0.sh
$SCREEN -S E0-GPU1 -dm bash /tmp/loco_e0_gpu1.sh
sleep 2
$SCREEN -ls | grep E0
