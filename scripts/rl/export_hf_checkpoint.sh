#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: export_hf_checkpoint.sh EXPERIMENT_DIR [STEP] [DTYPE] [BASE_MODEL_DIR]

Environment:
  VERL_ROOT       path to the verl checkout
  TARGET_DIR      optional output directory (default: EXPERIMENT_DIR/merged_hf_stepSTEP)
  DTYPE           optional dtype when the third positional argument is omitted
  BASE_MODEL_DIR  optional base model directory used to restore processor files

The experiment directory must contain global_step_<STEP>/actor. If STEP is
omitted, latest_checkpointed_iteration.txt is used.
EOF
    exit 2
}

[[ $# -ge 1 ]] || usage
: "${VERL_ROOT:?Set VERL_ROOT to the verl checkout}"
[[ -d "${VERL_ROOT}" ]] || { echo "verl directory not found: ${VERL_ROOT}" >&2; exit 2; }

EXP_DIR=$(realpath "$1")
STEP=${2:-}
DTYPE=${3:-${DTYPE:-bfloat16}}
BASE_MODEL_DIR=${4:-${BASE_MODEL_DIR:-}}

[[ -d "${EXP_DIR}" ]] || { echo "Experiment directory not found: ${EXP_DIR}" >&2; exit 2; }

if [[ -z "${STEP}" ]]; then
    LATEST_FILE="${EXP_DIR}/latest_checkpointed_iteration.txt"
    [[ -f "${LATEST_FILE}" ]] || {
        echo "Missing step and latest_checkpointed_iteration.txt: ${LATEST_FILE}" >&2
        exit 2
    }
    STEP=$(tr -d '[:space:]' < "${LATEST_FILE}")
fi

[[ "${STEP}" =~ ^[0-9]+$ ]] || { echo "STEP must be a non-negative integer: ${STEP}" >&2; exit 2; }
if [[ -n "${BASE_MODEL_DIR}" && ! -d "${BASE_MODEL_DIR}" ]]; then
    echo "Base model directory not found: ${BASE_MODEL_DIR}" >&2
    exit 2
fi

LOCAL_DIR="${EXP_DIR}/global_step_${STEP}/actor"
TARGET_DIR=${TARGET_DIR:-${EXP_DIR}/merged_hf_step${STEP}}

[[ -d "${LOCAL_DIR}" ]] || { echo "Actor checkpoint not found: ${LOCAL_DIR}" >&2; exit 2; }
if [[ -e "${TARGET_DIR}" ]]; then
    echo "Target already exists: ${TARGET_DIR}" >&2
    echo "Set TARGET_DIR to a new path rather than overwriting it." >&2
    exit 2
fi

export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"
python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${LOCAL_DIR}" \
    --target_dir "${TARGET_DIR}" \
    --trust-remote-code

[[ -f "${TARGET_DIR}/config.json" ]] || {
    echo "Export completed without config.json: ${TARGET_DIR}" >&2
    exit 2
}

python3 - "${TARGET_DIR}/config.json" "${DTYPE}" <<'PY'
import json
import sys

config_path, dtype = sys.argv[1:]
with open(config_path, encoding="utf-8") as handle:
    config = json.load(handle)


def update_dtype(node):
    if not isinstance(node, dict):
        return
    if "dtype" in node or "torch_dtype" in node:
        node["dtype"] = dtype
        node["torch_dtype"] = dtype
    for value in node.values():
        if isinstance(value, dict):
            update_dtype(value)


update_dtype(config)
with open(config_path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
PY

IS_MULTIMODAL=$(python3 - "${TARGET_DIR}/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print("1" if "vision_config" in json.load(handle) else "0")
PY
)

if [[ "${IS_MULTIMODAL}" == "1" ]]; then
    missing=()
    for filename in preprocessor_config.json processor_config.json; do
        [[ -f "${TARGET_DIR}/${filename}" ]] || missing+=("${filename}")
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        [[ -n "${BASE_MODEL_DIR}" ]] || {
            echo "The merged model is multimodal and needs BASE_MODEL_DIR to restore: ${missing[*]}" >&2
            exit 2
        }
        for filename in "${missing[@]}"; do
            [[ -f "${BASE_MODEL_DIR}/${filename}" ]] || {
                echo "Missing ${filename} in BASE_MODEL_DIR: ${BASE_MODEL_DIR}" >&2
                exit 2
            }
            cp "${BASE_MODEL_DIR}/${filename}" "${TARGET_DIR}/"
        done
    fi
fi

echo "Exported Hugging Face checkpoint: ${TARGET_DIR}"
