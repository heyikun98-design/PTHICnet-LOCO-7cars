#!/usr/bin/env bash
#
# 驗證 PT-HICNET 防禦性腳手架是否生效的自動化腳本
# 目的：故意觸發警報，驗證 fail-fast 與一致性檢查，而非追求跑通

set -Eeuo pipefail

# -----------------------------
# 基本路徑與全域參數
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-${PROJECT_ROOT}/configs/default.yaml}"
SEED="${SEED:-42}"
FILM_MODE="${FILM_MODE:-none}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_ROOT="${PROJECT_ROOT}/scripts/logs/verify_defenses/${TIMESTAMP}"
TMP_DIR="$(mktemp -d "${PROJECT_ROOT}/.verify_defenses_tmp.XXXXXX")"
BACKUP_CONFIG="${TMP_DIR}/config_backup.yaml"

PASS_COUNT=0
TOTAL_CASES=4

# Case1 產生出的 checkpoint（後續 case2~4 會重用）
CKPT_PATH=""

# -----------------------------
# 輸出工具函式
# -----------------------------
log_info() {
  echo "[INFO] $*"
}

log_pass() {
  echo "[PASS] $*"
}

log_fail() {
  echo "[FAIL] $*" >&2
}

# -----------------------------
# 清理與還原（無論成功失敗都會執行）
# -----------------------------
cleanup() {
  # 若備份存在，優先還原原始 config，避免污染工作區
  if [[ -f "${BACKUP_CONFIG}" ]]; then
    cp "${BACKUP_CONFIG}" "${CONFIG_PATH}"
  fi
  # 刪除暫存目錄
  if [[ -d "${TMP_DIR}" ]]; then
    rm -rf "${TMP_DIR}"
  fi
}
trap cleanup EXIT INT TERM

# -----------------------------
# 前置檢查
# -----------------------------
mkdir -p "${LOG_ROOT}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  log_fail "找不到 config：${CONFIG_PATH}"
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  log_fail "找不到 Python 執行檔：${PYTHON_BIN}"
  exit 1
fi

# 統一注入 PYTHONPATH，避免 train/eval 在不同啟動目錄下出現套件匯入失敗
# - PROJECT_ROOT: 讓 `models.*` 這類 package import 可解析
# - PROJECT_ROOT/models, PROJECT_ROOT/feather: 與既有腳本的 sys.path 習慣保持相容
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/models:${PROJECT_ROOT}/feather:${PYTHONPATH}"
else
  export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/models:${PROJECT_ROOT}/feather"
fi

# 檢查 PyYAML（本腳本需要用它做可逆 YAML 改寫）
if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import yaml  # noqa: F401
PY
then
  log_fail "目前 Python 環境缺少 PyYAML，無法改寫 config。"
  exit 1
fi

# 備份原始 config，後續每個 case 都可回復到乾淨狀態
cp "${CONFIG_PATH}" "${BACKUP_CONFIG}"

# -----------------------------
# YAML 工具函式（安全可逆）
# -----------------------------
restore_config() {
  cp "${BACKUP_CONFIG}" "${CONFIG_PATH}"
}

# 讀取 YAML 指定 key（dot-path）
yaml_get() {
  local file="$1"
  local key_path="$2"
  "${PYTHON_BIN}" - "$file" "$key_path" <<'PY'
import sys
import yaml

cfg_path, key_path = sys.argv[1], sys.argv[2]
with open(cfg_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

cur = data
for key in key_path.split("."):
    if not isinstance(cur, dict) or key not in cur:
        raise KeyError(f"missing key: {key_path}")
    cur = cur[key]

if isinstance(cur, (dict, list)):
    import json
    print(json.dumps(cur, ensure_ascii=False))
else:
    print(cur)
PY
}

# 設定 YAML 指定 key（value 使用 JSON 字串表示，避免型別誤判）
yaml_set_json() {
  local file="$1"
  local key_path="$2"
  local json_value="$3"
  "${PYTHON_BIN}" - "$file" "$key_path" "$json_value" <<'PY'
import json
import sys
import yaml

cfg_path, key_path, json_value = sys.argv[1], sys.argv[2], sys.argv[3]
with open(cfg_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

keys = key_path.split(".")
cur = data
for k in keys[:-1]:
    if k not in cur or not isinstance(cur[k], dict):
        raise KeyError(f"missing or non-dict path: {key_path}")
    cur = cur[k]

cur[keys[-1]] = json.loads(json_value)

with open(cfg_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
PY
}

json_quote() {
  local raw="$1"
  "${PYTHON_BIN}" - "$raw" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1], ensure_ascii=False))
PY
}

# -----------------------------
# 命令執行與斷言工具
# -----------------------------
run_expect_success() {
  local desc="$1"
  local log_file="$2"
  shift 2

  log_info "${desc}"
  set +e
  "$@" >"${log_file}" 2>&1
  local rc=$?
  set -e
  if [[ ${rc} -ne 0 ]]; then
    log_fail "${desc} 失敗（預期成功），exit=${rc}，log=${log_file}"
    return 1
  fi
  return 0
}

run_expect_fail() {
  local desc="$1"
  local log_file="$2"
  shift 2

  log_info "${desc}"
  set +e
  "$@" >"${log_file}" 2>&1
  local rc=$?
  set -e
  if [[ ${rc} -eq 0 ]]; then
    log_fail "${desc} 成功（預期失敗），防禦機制可能失效，log=${log_file}"
    return 1
  fi
  return 0
}

assert_file_exists() {
  local file_path="$1"
  local err_msg="$2"
  if [[ ! -f "${file_path}" ]]; then
    log_fail "${err_msg}（缺少檔案：${file_path}）"
    return 1
  fi
  return 0
}

assert_log_contains() {
  local log_file="$1"
  local pattern="$2"
  local err_msg="$3"
  if ! grep -E "${pattern}" "${log_file}" >/dev/null 2>&1; then
    log_fail "${err_msg}（pattern='${pattern}'，log=${log_file}）"
    return 1
  fi
  return 0
}

mark_case_pass() {
  local case_name="$1"
  PASS_COUNT=$((PASS_COUNT + 1))
  log_pass "${case_name} 通過 (${PASS_COUNT}/${TOTAL_CASES})"
}

# -----------------------------
# 推導 run 目錄（與 train 腳本命名規則一致）
# -----------------------------
EXP_NAME="$(yaml_get "${CONFIG_PATH}" "experiment.name")"
OUTPUT_ROOT="$(yaml_get "${CONFIG_PATH}" "experiment.output_root")"
RUN_NAME="${EXP_NAME}_seed${SEED}_film-${FILM_MODE}"
EXP_DIR="${PROJECT_ROOT}/${OUTPUT_ROOT}/${RUN_NAME}"
CKPT_CANDIDATE="${EXP_DIR}/checkpoints/best_model.pth"

# -----------------------------
# Case 1：正常短訓練（驗證 smoke test + warning + 落盤）
# -----------------------------
log_info "[CASE1] 正常 E2 單 Seed 短訓練"
restore_config
yaml_set_json "${CONFIG_PATH}" "training.epoch" "1"

CASE1_LOG="${LOG_ROOT}/case1_train.log"
run_expect_success \
  "[CASE1] 執行 train_pt_hicnet.py（epoch=1）" \
  "${CASE1_LOG}" \
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train_pt_hicnet.py" \
  --config "${CONFIG_PATH}" \
  --seed "${SEED}" \
  --film_mode "${FILM_MODE}"

assert_log_contains \
  "${CASE1_LOG}" \
  "category'.*UNUSED in the forward pass" \
  "[CASE1] 未偵測到 category 未使用警示"

assert_log_contains \
  "${CASE1_LOG}" \
  "\\[LossConfig\\] resolved delta=.*source=" \
  "[CASE1] 未偵測到 delta 來源解析 log"

assert_file_exists "${EXP_DIR}/config_used.yaml" "[CASE1] 未落盤 config_used.yaml"
assert_file_exists "${EXP_DIR}/config_hash.txt" "[CASE1] 未落盤 config_hash.txt"
assert_file_exists "${CKPT_CANDIDATE}" "[CASE1] 未產生 best_model.pth"

CKPT_PATH="${CKPT_CANDIDATE}"
mark_case_pass "CASE1"

# -----------------------------
# Case 2：篡改模型超參（預期 eval hard fail）
# -----------------------------
log_info "[CASE2] 篡改模型超參，驗證一致性 Hard Fail"
restore_config
yaml_set_json "${CONFIG_PATH}" "model.pt_nsample" "[31,32,32,32]"

CASE2_LOG="${LOG_ROOT}/case2_eval_model_mismatch.log"
run_expect_fail \
  "[CASE2] 執行 eval_pt_hicnet.py（模型超參篡改）" \
  "${CASE2_LOG}" \
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/eval_pt_hicnet.py" \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CKPT_PATH}" \
  --output "experiments/verify_defenses_case2.json"

assert_log_contains \
  "${CASE2_LOG}" \
  "Model hyperparameters mismatch|Checkpoint is the single source of truth|ValueError" \
  "[CASE2] 未捕捉到模型一致性錯誤訊息"

mark_case_pass "CASE2"

# -----------------------------
# Case 3：篡改訓練超參（預期 hash hard fail）
# -----------------------------
log_info "[CASE3] 篡改訓練超參，驗證 Hash Hard Fail"
restore_config
yaml_set_json "${CONFIG_PATH}" "training.learning_rate" "0.123456"

CASE3_LOG="${LOG_ROOT}/case3_eval_hash_mismatch.log"
run_expect_fail \
  "[CASE3] 執行 eval_pt_hicnet.py（訓練超參篡改）" \
  "${CASE3_LOG}" \
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/eval_pt_hicnet.py" \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CKPT_PATH}" \
  --output "experiments/verify_defenses_case3.json"

assert_log_contains \
  "${CASE3_LOG}" \
  "Config hash mismatch|ValueError" \
  "[CASE3] 未捕捉到 hash mismatch 錯誤訊息"

mark_case_pass "CASE3"

# -----------------------------
# Case 4：篡改 data_root（預期不觸發 hash fail）
# -----------------------------
log_info "[CASE4] 篡改 data_root，驗證 Hash 白名單忽略無關配置"
restore_config
ORIG_DATA_ROOT="$(yaml_get "${CONFIG_PATH}" "data.data_root")"

# 使用等價路徑（原路徑 + '/.'）避免破壞實際資料可讀性
if [[ "${ORIG_DATA_ROOT}" == */ ]]; then
  NEW_DATA_ROOT="${ORIG_DATA_ROOT}."
else
  NEW_DATA_ROOT="${ORIG_DATA_ROOT}/."
fi
yaml_set_json "${CONFIG_PATH}" "data.data_root" "$(json_quote "${NEW_DATA_ROOT}")"

CASE4_LOG="${LOG_ROOT}/case4_eval_data_root_changed.log"
run_expect_success \
  "[CASE4] 執行 eval_pt_hicnet.py（僅 data_root 變更）" \
  "${CASE4_LOG}" \
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/eval_pt_hicnet.py" \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CKPT_PATH}" \
  --output "experiments/verify_defenses_case4.json"

assert_log_contains \
  "${CASE4_LOG}" \
  "hash verified" \
  "[CASE4] 未看到 hash verified，請檢查是否非預期路徑觸發錯誤"

mark_case_pass "CASE4"

# -----------------------------
# 收尾總結
# -----------------------------
restore_config

echo
if [[ ${PASS_COUNT} -eq ${TOTAL_CASES} ]]; then
  log_pass "全部測試完成：${PASS_COUNT}/${TOTAL_CASES} 通過"
  log_info "完整 log 請查看：${LOG_ROOT}"
  exit 0
else
  log_fail "測試未全數通過：${PASS_COUNT}/${TOTAL_CASES}"
  log_info "完整 log 請查看：${LOG_ROOT}"
  exit 1
fi
