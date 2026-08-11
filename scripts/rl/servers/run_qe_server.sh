#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

QE_MODEL_PATH=${QE_MODEL_PATH:-}
if [[ -z "${QE_MODEL_PATH}" ]]; then
    echo "Missing required environment variable: QE_MODEL_PATH" >&2
    exit 2
fi
if [[ ! -e "${QE_MODEL_PATH}" ]]; then
    echo "COMET model path not found: ${QE_MODEL_PATH}" >&2
    exit 2
fi

OPENLID_MODEL_PATH=${OPENLID_MODEL_PATH:-}
if [[ -n "${OPENLID_MODEL_PATH}" && ! -f "${OPENLID_MODEL_PATH}" ]]; then
    echo "OpenLID model not found: ${OPENLID_MODEL_PATH}" >&2
    exit 2
fi

QE_HOST=${QE_HOST:-127.0.0.1}
QE_PORT=${QE_PORT:-8008}
QE_DEVICE=${QE_DEVICE:-cuda:0}
QE_PREDICT_BATCH_SIZE=${QE_PREDICT_BATCH_SIZE:-32}
QE_MAX_SERVER_BATCH_SIZE=${QE_MAX_SERVER_BATCH_SIZE:-32}
QE_MAX_WAIT_MS=${QE_MAX_WAIT_MS:-20}
QE_OPENLID_MAX_WAIT_MS=${QE_OPENLID_MAX_WAIT_MS:-20}
QE_OPENLID_MAX_BATCH_SIZE=${QE_OPENLID_MAX_BATCH_SIZE:-64}
QE_SLOW_REQUEST_MS=${QE_SLOW_REQUEST_MS:-3000}
QE_LOG_LEVEL=${QE_LOG_LEVEL:-info}
QE_HF_CACHE=${QE_HF_CACHE:-${HF_HOME:-}}
QE_SET_HF_HUB_CACHE=${QE_SET_HF_HUB_CACHE:-1}
QE_LOCAL_FILES_ONLY=${QE_LOCAL_FILES_ONLY:-1}

ARGS=(
    --model "${QE_MODEL_PATH}"
    --host "${QE_HOST}"
    --port "${QE_PORT}"
    --device "${QE_DEVICE}"
    --predict-batch-size "${QE_PREDICT_BATCH_SIZE}"
    --max-server-batch-size "${QE_MAX_SERVER_BATCH_SIZE}"
    --max-wait-ms "${QE_MAX_WAIT_MS}"
    --openlid-max-wait-ms "${QE_OPENLID_MAX_WAIT_MS}"
    --openlid-max-batch-size "${QE_OPENLID_MAX_BATCH_SIZE}"
    --slow-request-ms "${QE_SLOW_REQUEST_MS}"
    --log-level "${QE_LOG_LEVEL}"
)

if [[ -n "${OPENLID_MODEL_PATH}" ]]; then
    ARGS+=(--openlid-model "${OPENLID_MODEL_PATH}")
fi
if [[ -n "${QE_HF_CACHE}" ]]; then
    ARGS+=(--hf-cache "${QE_HF_CACHE}")
fi
if [[ "${QE_SET_HF_HUB_CACHE}" == "0" ]]; then
    unset HUGGINGFACE_HUB_CACHE || true
    ARGS+=(--no-hf-hub-cache)
fi
if [[ "${QE_LOCAL_FILES_ONLY}" == "0" ]]; then
    ARGS+=(--no-local-files-only)
fi

export PYTHONUNBUFFERED=1
exec python3 "${SCRIPT_DIR}/qe_server.py" "${ARGS[@]}"
