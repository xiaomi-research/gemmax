#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

XCOMET_MODEL_PATH=${XCOMET_MODEL_PATH:-}
COMETKIWI_MODEL_PATH=${COMETKIWI_MODEL_PATH:-}
OPENLID_MODEL_PATH=${OPENLID_MODEL_PATH:-}

for required in XCOMET_MODEL_PATH COMETKIWI_MODEL_PATH OPENLID_MODEL_PATH; do
    if [[ -z "${!required}" ]]; then
        echo "Missing required environment variable: ${required}" >&2
        exit 2
    fi
done

XCOMET_DEVICES=${XCOMET_DEVICES:-0}
COMETKIWI_DEVICES=${COMETKIWI_DEVICES:-1}
XCOMET_BASE_PORT=${XCOMET_BASE_PORT:-8008}
COMETKIWI_BASE_PORT=${COMETKIWI_BASE_PORT:-8012}
REWARD_HOST=${REWARD_HOST:-0.0.0.0}
REWARD_ADVERTISE_HOST=${REWARD_ADVERTISE_HOST:-127.0.0.1}

IFS=',' read -r -a XCOMET_DEVICE_LIST <<< "${XCOMET_DEVICES}"
IFS=',' read -r -a COMETKIWI_DEVICE_LIST <<< "${COMETKIWI_DEVICES}"
if [[ ${#XCOMET_DEVICE_LIST[@]} -eq 0 || ${#COMETKIWI_DEVICE_LIST[@]} -eq 0 ]]; then
    echo "Both XCOMET_DEVICES and COMETKIWI_DEVICES must list at least one GPU." >&2
    exit 2
fi
XCOMET_LAST_PORT=$((XCOMET_BASE_PORT + ${#XCOMET_DEVICE_LIST[@]} - 1))
COMETKIWI_LAST_PORT=$((COMETKIWI_BASE_PORT + ${#COMETKIWI_DEVICE_LIST[@]} - 1))
if (( XCOMET_BASE_PORT <= COMETKIWI_LAST_PORT && COMETKIWI_BASE_PORT <= XCOMET_LAST_PORT )); then
    echo "xCOMET and CometKiwi port ranges overlap; adjust XCOMET_BASE_PORT or COMETKIWI_BASE_PORT." >&2
    exit 2
fi

XCOMET_PREDICT_BATCH_SIZE=${XCOMET_PREDICT_BATCH_SIZE:-32}
XCOMET_MAX_SERVER_BATCH_SIZE=${XCOMET_MAX_SERVER_BATCH_SIZE:-32}
COMETKIWI_PREDICT_BATCH_SIZE=${COMETKIWI_PREDICT_BATCH_SIZE:-32}
COMETKIWI_MAX_SERVER_BATCH_SIZE=${COMETKIWI_MAX_SERVER_BATCH_SIZE:-32}
XCOMET_SET_HF_HUB_CACHE=${XCOMET_SET_HF_HUB_CACHE:-1}
COMETKIWI_SET_HF_HUB_CACHE=${COMETKIWI_SET_HF_HUB_CACHE:-1}

PIDS=()
XCOMET_ENDPOINTS=()
COMETKIWI_ENDPOINTS=()

cleanup() {
    trap - EXIT INT TERM
    for pid in "${PIDS[@]:-}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill -TERM "${pid}" 2>/dev/null || true
        fi
    done
    wait "${PIDS[@]:-}" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for index in "${!XCOMET_DEVICE_LIST[@]}"; do
    device=${XCOMET_DEVICE_LIST[$index]// /}
    port=$((XCOMET_BASE_PORT + index))
    CUDA_VISIBLE_DEVICES="${device}" \
        QE_MODEL_PATH="${XCOMET_MODEL_PATH}" \
        OPENLID_MODEL_PATH="${OPENLID_MODEL_PATH}" \
        QE_HOST="${REWARD_HOST}" \
        QE_PORT="${port}" \
        QE_DEVICE=cuda:0 \
        QE_PREDICT_BATCH_SIZE="${XCOMET_PREDICT_BATCH_SIZE}" \
        QE_MAX_SERVER_BATCH_SIZE="${XCOMET_MAX_SERVER_BATCH_SIZE}" \
        QE_SET_HF_HUB_CACHE="${XCOMET_SET_HF_HUB_CACHE}" \
        bash "${SCRIPT_DIR}/run_qe_server.sh" &
    PIDS+=("$!")
    XCOMET_ENDPOINTS+=("http://${REWARD_ADVERTISE_HOST}:${port}/score_openlid")
done

for index in "${!COMETKIWI_DEVICE_LIST[@]}"; do
    device=${COMETKIWI_DEVICE_LIST[$index]// /}
    port=$((COMETKIWI_BASE_PORT + index))
    CUDA_VISIBLE_DEVICES="${device}" \
        QE_MODEL_PATH="${COMETKIWI_MODEL_PATH}" \
        OPENLID_MODEL_PATH= \
        QE_HOST="${REWARD_HOST}" \
        QE_PORT="${port}" \
        QE_DEVICE=cuda:0 \
        QE_PREDICT_BATCH_SIZE="${COMETKIWI_PREDICT_BATCH_SIZE}" \
        QE_MAX_SERVER_BATCH_SIZE="${COMETKIWI_MAX_SERVER_BATCH_SIZE}" \
        QE_SET_HF_HUB_CACHE="${COMETKIWI_SET_HF_HUB_CACHE}" \
        bash "${SCRIPT_DIR}/run_qe_server.sh" &
    PIDS+=("$!")
    COMETKIWI_ENDPOINTS+=("http://${REWARD_ADVERTISE_HOST}:${port}/score")
done

printf -v XCOMET_URLS '%s,' "${XCOMET_ENDPOINTS[@]}"
printf -v COMETKIWI_URLS '%s,' "${COMETKIWI_ENDPOINTS[@]}"
echo "Reward servers started. Use these training variables:"
echo "XCOMET_URLS=${XCOMET_URLS%,}"
echo "COMETKIWI_URLS=${COMETKIWI_URLS%,}"

set +e
wait -n "${PIDS[@]}"
status=$?
set -e
if [[ ${status} -eq 0 ]]; then
    status=1
fi
echo "A reward server exited; stopping the remaining processes." >&2
exit "${status}"
