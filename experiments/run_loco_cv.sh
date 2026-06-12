#!/bin/bash
# LOCO-CV: Leave-One-Car-Out Cross-Validation (7 folds)
# Usage: bash experiments/run_loco_cv.sh [smoke]
#   smoke = run only the first fold as a smoke test

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON=/data1/user/yikun/.conda/envs/dl/bin/python
SCREEN=/data1/user/yikun/.conda/envs/dl/bin/screen

# 7 vehicles = 7 folds
VEHICLES=(C201 EP32 JX65 CY02C M6 S50EVK FX11)
CAR_DIRS=(car1 car2 car3 car4 car5 car6 car7)

SMOKE_ONLY="${1:-}"

# GPU assignment: fold 0-3 on GPU0, fold 4-6 on GPU1
GPU0_FOLDS=(0 1 2 3)
GPU1_FOLDS=(4 5 6)

run_fold() {
    local fold="$1"
    local vehicle="${VEHICLES[$fold]}"
    local car_dir="${CAR_DIRS[$fold]}"
    local gpu="$2"
    local exp_name="pt_hicnet_loco_fold${fold}_${vehicle}"

    echo "[LOCO] Fold ${fold}: held-out=${vehicle} (${car_dir}) on GPU ${gpu}"

    local tmp_script
    tmp_script="/tmp/loco_fold${fold}_${vehicle}.sh"
    cat > "$tmp_script" << SCRIPT
#!/bin/bash
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=${gpu}
${PYTHON} -u scripts/train_pt_hicnet.py \\
  --config configs/default.yaml \\
  --film_mode global \\
  --patience 50 \\
  --restore_best \\
  --seed 42 \\
  --test_vehicles "${vehicle}" \\
  --val_split 0.15 \\
  --split_seed 2026 \\
  --exp_name "${exp_name}"
SCRIPT
    chmod +x "$tmp_script"
    $SCREEN -S "LOCO-F${fold}_${vehicle}" -dm "$tmp_script"
}

# --- Main ---
echo "=== LOCO-CV Launcher ==="
echo ""

RUN_FOLDS=("${!VEHICLES[@]}")
if [ "$SMOKE_ONLY" = "smoke" ]; then
    echo "[Smoke] Running only fold 0 (C201 / car1)"
    RUN_FOLDS=(0)
fi

# Launch GPU0 folds
for fold in "${GPU0_FOLDS[@]}"; do
    if [[ " ${RUN_FOLDS[*]} " =~ " ${fold} " ]]; then
        run_fold "$fold" 0
    fi
done

# Launch GPU1 folds
for fold in "${GPU1_FOLDS[@]}"; do
    if [[ " ${RUN_FOLDS[*]} " =~ " ${fold} " ]]; then
        run_fold "$fold" 1
    fi
done

sleep 2
echo ""
echo "Launched screens:"
$SCREEN -ls | grep LOCO

echo ""
echo "[LOCO] Monitor with: screen -r LOCO-F<N>_<vehicle>"
echo "[LOCO] After completion, run: ${PYTHON} scripts/aggregate_loco.py"
