# GRPO Fine-tuning and Model Merging

This directory contains the code used for the MiLMMT-46 reinforcement-learning workflow:

- `run_grpo.sh`: GRPO training with `verl` and vLLM.
- `rewards/mt_dual_comet_reward.py`: dual-QE reward with an OpenLID language gate.
- `servers/qe_server.py`: batched COMET QE service with an optional OpenLID-gated endpoint.
- `servers/run_qe_server.sh`: launch one xCOMET or CometKiwi service process.
- `servers/start_reward_servers.sh`: launch xCOMET/OpenLID and CometKiwi GPU pools.
- `export_hf_checkpoint.sh`: convert a sharded `verl` FSDP actor checkpoint to Hugging Face format.
- `linear_merge.py`: linearly interpolate the SFT and RL Hugging Face checkpoints.

## Environment

Python 3.10 or later is required. The released runs used `verl` commit
`8ebf167e32e790e92a85eae2ddbecb8c515d8156`.

```bash
git clone https://github.com/volcengine/verl.git /path/to/verl
git -C /path/to/verl checkout 8ebf167e32e790e92a85eae2ddbecb8c515d8156

pip install -r requirements-rl.txt
pip install -e "/path/to/verl[vllm]"
```

PyTorch and vLLM wheels are CUDA-dependent. Adjust their installation when the pinned versions in
`requirements-rl.txt` do not match your CUDA and driver environment.

## Input Data

`run_grpo.sh` consumes existing `verl`-compatible Parquet files. It does not create, split, filter, or score data.
`examples/rl.json` is a small, human-readable JSON example of the required row structure; it is not passed directly
to `run_grpo.sh`. Convert your JSON records into separate `train.parquet` and `val.parquet` files before training.

For example, using the `datasets` library:

```bash
python3 - <<'PY'
import json
from pathlib import Path

from datasets import Dataset

rows = json.loads(Path("examples/rl.json").read_text())
for split in ("train", "val"):
    split_rows = [row for row in rows if row["extra_info"]["split"] == split]
    if not split_rows:
        raise ValueError(f"No rows found for split={split!r}")
    Dataset.from_list(split_rows).to_parquet(f"{split}.parquet")
PY
```

Then pass the generated files through `TRAIN_FILE` and `VAL_FILE`. Each row should contain the standard `verl` RL fields and, for the included reward, at least:

```text
prompt: list[{"role": "user", "content": "..."}]
reward_model.ground_truth: reference translation
extra_info.source: source sentence
extra_info.target_language: expected target-language name
```

The target-language name must use the naming convention expected by the OpenLID service.

## Reward Services

The included server uses direct COMET `predict_step` inference and batches concurrent HTTP requests before each
model forward pass. The same implementation serves either xCOMET or CometKiwi, depending on the checkpoint passed
through `QE_MODEL_PATH`. Supplying `OPENLID_MODEL_PATH` additionally enables the OpenLID-gated route.

Prepare local paths to the following model files before starting the services:

- an xCOMET checkpoint directory or `.ckpt` file;
- a CometKiwi checkpoint directory or `.ckpt` file;
- an OpenLID-v3 fastText `.bin` model.

Model files are not stored in this repository. If a COMET directory is passed, the server first looks for
`checkpoints/model.ckpt` and then for another `.ckpt` below that directory.

### Start Individual Services

Start xCOMET with OpenLID on port 8008:

```bash
CUDA_VISIBLE_DEVICES=0 \
QE_MODEL_PATH=/path/to/xcomet \
OPENLID_MODEL_PATH=/path/to/openlid-v3.bin \
QE_PORT=8008 \
bash scripts/rl/servers/run_qe_server.sh
```

Start CometKiwi on port 8012:

```bash
CUDA_VISIBLE_DEVICES=1 \
QE_MODEL_PATH=/path/to/cometkiwi \
QE_PORT=8012 \
bash scripts/rl/servers/run_qe_server.sh
```

Set `QE_HOST` to the desired bind address. The main tuning variables are
`QE_PREDICT_BATCH_SIZE`, `QE_MAX_SERVER_BATCH_SIZE`, `QE_MAX_WAIT_MS`, and `QE_DEVICE`. Set
`QE_SET_HF_HUB_CACHE=0` for checkpoints whose encoder configuration contains a local absolute path that must not be
resolved through the Hugging Face cache.

### Start Both GPU Pools

The convenience launcher starts one process per listed GPU and prints the exact URL lists to pass to training:

```bash
XCOMET_MODEL_PATH=/path/to/xcomet \
COMETKIWI_MODEL_PATH=/path/to/cometkiwi \
OPENLID_MODEL_PATH=/path/to/openlid-v3.bin \
XCOMET_DEVICES=0,1,2,3 \
COMETKIWI_DEVICES=4,5,6,7 \
bash scripts/rl/servers/start_reward_servers.sh
```

xCOMET/OpenLID ports start at `XCOMET_BASE_PORT=8008`; CometKiwi ports start at
`COMETKIWI_BASE_PORT=8012`. Override `REWARD_ADVERTISE_HOST` when the printed URLs should use a hostname or address
other than `127.0.0.1`.

### API Contract

The training script accepts one or more comma-separated endpoint URLs:

- `XCOMET_URLS`: full `/score_openlid` endpoints for xCOMET QE plus OpenLID gating.
- `COMETKIWI_URLS`: full `/score` endpoints for CometKiwi QE.

The xCOMET/OpenLID endpoint receives:

```json
{"items": [{"src": "source", "mt": "translation", "tgt_lang": "English"}]}
```

and returns at least the fields consumed by the reward client:

```json
{
  "results": [{
    "qe": 0.82,
    "la_ok": 1,
    "la_skip": 0,
    "pred_iso": "eng_Latn",
    "tgt_iso": "eng_Latn"
  }]
}
```

The CometKiwi endpoint receives `{"items": [{"src": "source", "mt": "translation"}]}` and returns at least
`{"results": [{"score": 0.79}]}`.

When `la_ok=1`, the reward is `(xCOMET + CometKiwi) / 2`. When `la_ok=0`, the reward is
`DUALCOMET_LA_FAIL_SCORE`, which defaults to `0.0`. This preserves the released OpenLID hard-gating behavior.
`XCOMET_RETRIES=-1` enables infinite retries; set a non-negative value to fail after a bounded number of retries.

Check either service with `curl http://127.0.0.1:8008/health`. A pure QE request can be tested with:

```bash
curl -X POST http://127.0.0.1:8012/score \
  -H 'Content-Type: application/json' \
  -d '{"items":[{"src":"Hello","mt":"Bonjour"}]}'
```

## GRPO Training

The launcher defaults reproduce the main 12B training hyperparameters: rollout `n=8`, prompt/response limits of
4096 tokens, learning rate `1e-6`, KL coefficient `0.001`, batch and mini-batch sizes of 128, and three epochs.
Hardware, paths, and run names remain configurable through environment variables.

```bash
VERL_ROOT=/path/to/verl \
MODEL_PATH=xiaomi-research/MiLMMT-46-12B-v0.1 \
TRAIN_FILE=/path/to/train.parquet \
VAL_FILE=/path/to/val.parquet \
XCOMET_URLS=http://reward-host-1:8008/score_openlid,http://reward-host-2:8008/score_openlid \
COMETKIWI_URLS=http://reward-host-1:8012/score,http://reward-host-2:8012/score \
NNODES=4 \
NGPUS_PER_NODE=8 \
bash scripts/rl/run_grpo.sh
```

The default logger is console-only. To enable Weights & Biases, export credentials in the environment rather than
putting them in the script or command-line overrides:

```bash
export WANDB_API_KEY=...
export WANDB_ENTITY=...
export LOGGER='["console","wandb"]'
```

Then run the same training command above. Do not put API keys directly in scripts or Hydra command-line overrides.

For distributed runs, follow the standard `verl` launch procedure for your environment before starting training.

## Export a Hugging Face Checkpoint

The exporter reads `latest_checkpointed_iteration.txt` when `STEP` is omitted and expects the actor shards under
`global_step_<STEP>/actor`.

```bash
VERL_ROOT=/path/to/verl \
BASE_MODEL_DIR=/path/to/MiLMMT-46-12B-v0.1 \
bash scripts/rl/export_hf_checkpoint.sh /path/to/experiment
```

The default output is `/path/to/experiment/merged_hf_step<STEP>`. Set `TARGET_DIR` to choose another directory.
`BASE_MODEL_DIR` is only used to restore missing processor configuration files for multimodal Gemma 3 checkpoints.
The exporter refuses to overwrite an existing output directory.

## Merge SFT and RL Weights

Both inputs must already be Hugging Face safetensors checkpoints with the same architecture. `model_a` is the SFT
checkpoint and receives weight `alpha`; `model_b` is the exported RL checkpoint and receives weight `1 - alpha`:

```text
theta_merged = alpha * theta_SFT + (1 - alpha) * theta_RL
```

For an equal interpolation:

```bash
python3 scripts/rl/linear_merge.py \
  --model_a /path/to/MiLMMT-46-12B-v0.1 \
  --model_b /path/to/experiment/merged_hf_step714 \
  --alpha 0.5 \
  --out /path/to/MiLMMT-46-12B-v1.0
```

The output follows the RL checkpoint's shard layout and metadata and includes `merge_manifest.json`. The script
rejects `alpha` values outside `[0, 1]` and refuses to overwrite an existing output directory unless
`--overwrite` is supplied.
