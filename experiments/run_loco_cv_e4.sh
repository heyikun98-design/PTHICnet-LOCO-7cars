#!/bin/bash
# E4 LOCO-CV: Point Transformer + EarlyFusion + FiLM deep
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON=/data1/user/yikun/.conda/envs/dl/bin/python
SCREEN=/data1/user/yikun/.conda/envs/dl/bin/screen
echo "=== E4 LOCO-CV (FiLM deep) ==="

cat > /tmp/loco_e4_gpu0.sh << 'INNER'
#!/bin/bash
cd /data1/user/yikun/project/PT-HICNET
PYTHON=/data1/user/yikun/.conda/envs/dl/bin/python
for pair in "0 C201" "1 EP32" "2 JX65" "3 CY02C"; do
  fold=${pair% *}; vehicle=${pair#* }
  echo "=== E4 Fold ${fold}: ${vehicle} ==="
  CUDA_VISIBLE_DEVICES=0 ${PYTHON} -u scripts/train_pt_hicnet.py \
    --config configs/default.yaml --film_mode deep \
    --patience 50 --restore_best --seed 42 \
    --test_vehicles "${vehicle}" --val_split 0.15 --split_seed 2026 \
    --exp_name "pt_hicnet_loco_e4_fold${fold}_${vehicle}_seed42"
  echo "=== E4 Fold ${fold} done ==="
done
INNER

cat > /tmp/loco_e4_gpu1.sh << 'INNER'
#!/bin/bash
cd /data1/user/yikun/project/PT-HICNET
PYTHON=/data1/user/yikun/.conda/envs/dl/bin/python
for pair in "4 M6" "5 S50EVK" "6 FX11"; do
  fold=${pair% *}; vehicle=${pair#* }
  echo "=== E4 Fold ${fold}: ${vehicle} ==="
  CUDA_VISIBLE_DEVICES=1 ${PYTHON} -u scripts/train_pt_hicnet.py \
    --config configs/default.yaml --film_mode deep \
    --patience 50 --restore_best --seed 42 \
    --test_vehicles "${vehicle}" --val_split 0.15 --split_seed 2026 \
    --exp_name "pt_hicnet_loco_e4_fold${fold}_${vehicle}_seed42"
  echo "=== E4 Fold ${fold} done ==="
done
INNER

chmod +x /tmp/loco_e4_gpu0.sh /tmp/loco_e4_gpu1.sh
$SCREEN -S E4-GPU0 -dm bash /tmp/loco_e4_gpu0.sh
$SCREEN -S E4-GPU1 -dm bash /tmp/loco_e4_gpu1.sh
sleep 2
$SCREEN -ls | grep E4
