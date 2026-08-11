#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

VERL_ROOT=${VERL_ROOT:-}
MODEL_PATH=${MODEL_PATH:-}
TRAIN_FILE=${TRAIN_FILE:-}
VAL_FILE=${VAL_FILE:-}
XCOMET_URLS=${XCOMET_URLS:-}
COMETKIWI_URLS=${COMETKIWI_URLS:-}

for required in VERL_ROOT MODEL_PATH TRAIN_FILE VAL_FILE XCOMET_URLS COMETKIWI_URLS; do
    if [[ -z "${!required}" ]]; then
        echo "Missing required environment variable: ${required}" >&2
        exit 2
    fi
done

[[ -f "${TRAIN_FILE}" ]] || { echo "Training file not found: ${TRAIN_FILE}" >&2; exit 2; }
[[ -f "${VAL_FILE}" ]] || { echo "Validation file not found: ${VAL_FILE}" >&2; exit 2; }
[[ -d "${VERL_ROOT}" ]] || { echo "verl directory not found: ${VERL_ROOT}" >&2; exit 2; }
if [[ "${MODEL_PATH}" == /* && ! -d "${MODEL_PATH}" ]]; then
    echo "Local model directory not found: ${MODEL_PATH}" >&2
    exit 2
fi

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-8192}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.8}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.95}
ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS:-16}
ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
SAVE_FREQ=${SAVE_FREQ:-238}
TEST_FREQ=${TEST_FREQ:-50}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-1000}
FILTER_OVERLONG_PROMPTS_WORKERS=${FILTER_OVERLONG_PROMPTS_WORKERS:-16}
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-8}

XCOMET_TIMEOUT_S=${XCOMET_TIMEOUT_S:-300}
XCOMET_RETRIES=${XCOMET_RETRIES:--1}
DUALCOMET_LA_FAIL_SCORE=${DUALCOMET_LA_FAIL_SCORE:-0.0}
XCOMET_NO_KEEPALIVE=${XCOMET_NO_KEEPALIVE:-1}

PROJECT_NAME=${PROJECT_NAME:-milmmt_grpo_mt}
MODEL_SIZE_TAG=$(
    basename "${MODEL_PATH}" \
        | grep -oiE '[0-9]+\.?[0-9]*[bB]' \
        | head -n1 \
        | tr '[:upper:]' '[:lower:]' \
        || true
)
MODEL_SIZE_TAG=${MODEL_SIZE_TAG:-unknown}
DATA_TAG=$(basename "$(dirname "${TRAIN_FILE}")")
EXPERIMENT_NAME=${EXPERIMENT_NAME:-milmmt_${MODEL_SIZE_TAG}_grpo_dualcometopenlid_${DATA_TAG}_$(date +%Y%m%d_%H%M)}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/outputs/${EXPERIMENT_NAME}}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${PROJECT_DIR}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}
LOGGER=${LOGGER:-'["console"]'}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-${PROJECT_DIR}/outputs/tensorboard}
PLAIN_CHAT_TEMPLATE=${PLAIN_CHAT_TEMPLATE:-"{% for message in messages %}{{ message.content }}{% endfor %}"}
REWARD_PATH=${REWARD_PATH:-${SCRIPT_DIR}/rewards/mt_dual_comet_reward.py}
[[ -f "${REWARD_PATH}" ]] || { echo "Reward function not found: ${REWARD_PATH}" >&2; exit 2; }

export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"
export TENSORBOARD_DIR
export XCOMET_URLS COMETKIWI_URLS XCOMET_TIMEOUT_S XCOMET_RETRIES XCOMET_NO_KEEPALIVE
export HF_HOME=${HF_HOME:-${PROJECT_DIR}/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${PROJECT_DIR}/.cache}
export WANDB_DIR=${WANDB_DIR:-${PROJECT_DIR}/wandb}

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${XDG_CACHE_HOME}/verl" \
    "${PROJECT_DIR}/outputs" "${CHECKPOINT_DIR}"

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${VAL_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.train_max_samples=${TRAIN_MAX_SAMPLES}
    data.val_max_samples=${VAL_MAX_SAMPLES}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers=${FILTER_OVERLONG_PROMPTS_WORKERS}
    data.truncation=error
    data.shuffle=True
    data.seed=42
    "+data.cache_dir=${XDG_CACHE_HOME}/verl/rlhf"
    "+data.apply_chat_template_kwargs.chat_template='${PLAIN_CHAT_TEMPLATE}'"
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    "actor_rollout_ref.model.custom_chat_template='${PLAIN_CHAT_TEMPLATE}'"
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}
    actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.agent.num_workers=${ROLLOUT_AGENT_NUM_WORKERS}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD}
)

REWARD=(
    reward.num_workers=${REWARD_NUM_WORKERS}
    reward.reward_manager.name=naive
    reward.custom_reward_function.path="${REWARD_PATH}"
    reward.custom_reward_function.name=compute_score
    "+reward.custom_reward_function.reward_kwargs.xcomet_urls='${XCOMET_URLS}'"
    "+reward.custom_reward_function.reward_kwargs.cometkiwi_urls='${COMETKIWI_URLS}'"
    "+reward.custom_reward_function.reward_kwargs.timeout_s=${XCOMET_TIMEOUT_S}"
    "+reward.custom_reward_function.reward_kwargs.retries=${XCOMET_RETRIES}"
    "+reward.custom_reward_function.reward_kwargs.la_fail_score=${DUALCOMET_LA_FAIL_SCORE}"
)

TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.logger="${LOGGER}"
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.val_before_train=${VAL_BEFORE_TRAIN}
    trainer.default_local_dir="${CHECKPOINT_DIR}"
)

RAY_ENV=()
if [[ -n "${XCOMET_NO_KEEPALIVE}" ]]; then
    RAY_ENV+=(
        "+ray_kwargs.ray_init.runtime_env.env_vars.XCOMET_NO_KEEPALIVE='${XCOMET_NO_KEEPALIVE}'"
    )
fi

cd "${VERL_ROOT}"
python3 -m verl.trainer.main_ppo \
    hydra.run.dir="${OUTPUT_DIR}" \
    hydra.job.chdir=False \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${RAY_ENV[@]}" \
    "$@"
