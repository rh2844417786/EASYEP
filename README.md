# Domain Specific Pruning of Large Mixture-of-Experts Models with Few-shot Demonstrations ([📃 Paper](https://arxiv.org/abs/2504.06792))

## Table of Contents
- [1. Introduction](#1-introduction)
- [2. Preparation](#2-preparation)
- [3. Expert Selection](#3-expert-selection)
- [4. Model Pruning](#4-model-pruning)
- [5. Evaluation](#5-evaluation)
- [6. Citation](#6-citation)

> [!IMPORTANT]
> DeepSeek-V4-Flash is not compatible with the repository's vendored SGLang
> 0.4.3 / DeepSeek-V2 patch. For the diagnosed H100 baseline, V4 probe and the
> remaining hash-routing limitation, read
> [`docs/deepseek_v4_flash_reproduction.md`](docs/deepseek_v4_flash_reproduction.md).
> The full/prune-25/prune-50 accuracy, wall-time, and GPU-memory matrix is
> documented in
> [`docs/easyep_reproduction_matrix.md`](docs/easyep_reproduction_matrix.md).

## 1. Introduction
Mixture-of-Experts (MoE) models achieve a favorable trade-off between performance and inference efficiency by activating only a subset of experts. However, the memory overhead of storing all experts remains a major limitation, especially in large-scale MoE models such as DeepSeek-R1 (671B). In this study, we investigate domain specialization and expert redundancy in large-scale MoE models and uncover a consistent behavior we term *few-shot expert localization*, with only a few demonstrations, the model consistently activates a sparse and stable subset of experts. Building on this observation, we propose a simple yet effective pruning framework, **EASY-EP**, that leverages a few domain-specific demonstrations to identify and retain only the most relevant experts. EASY-EP comprises two key components: **output-aware expert importance assessment** and **expert-level token contribution estimation**. The former evaluates the importance of each expert for the current token by considering the gating scores and magnitudes of the outputs of activated experts, while the latter assesses the contribution of tokens based on representation similarities after and before routed experts. Experiments show that our method can achieve comparable performances and $2.99\times$ throughput under the same memory budget with full DeepSeek-R1 with only half the experts.

![](framework-v5.png)

## 2. Preparation
---

### 2.1 Requirements
```bash
cd EASYEP
conda create -n easyep python=3.10
conda activate easyep
pip install -r requirements.txt
```
### 2.2 Model Preparation

#### System Requirements
> [!NOTE] 
> Linux with Python 3.10 only. Mac and Windows are not supported.

Dependencies:
```pip-requirements
torch==2.4.1
triton==3.0.0
transformers==4.46.3
safetensors==0.4.5
```

#### Model Weights Conversion
Our code is based on the official inference demo provided by [DeepSeek](https://github.com/deepseek-ai/DeepSeek-V3/tree/main?tab=readme-ov-file#61-inference-with-deepseek-infer-demo-example-only). It requires Model Weights Conversion. Here, using an 8 x H200 141GB node, the conversion method is as follows:

Download the model weights from Hugging Face, and put them into /path/to/DeepSeek-R1 folder. Then convert Hugging Face model weights to a specific format:

```shell
python pruning/convert.py --hf-ckpt-path /path/to/DeepSeek-R1 --save-path /path/to/DeepSeek-R1-Demo --n-experts 256 --model-parallel 8
```

### Data Preparation


## 3. Expert Selection
---

This part primarily involves extracting and calculating internal MoE hidden states using calibration data, which will be used for later pruning.
```bash
torchrun --nproc_per_node=8 pruning/inf_new.py \
    --ckpt-path /path/to/DeepSeek-R1-Demo \
    --config configs/config_671B.json \
    --input-file dataset/aime23_full \
    --output expert_statistics/token_information/aime.jsonl
```

The expert mask matrix is then derived using statistical information. 
```bash
python pruning/expert_selection.py \
    --input_file expert_statistics/token_information/aime.jsonl \
    --output_file expert_statistics/expert_information/aime23.pt \
    --expert_mask expert_statistics/expert_mask/aime23_128_mask.json \
    --target_number 128 \
    --num-experts 256
```
For mixed-domain pruning, we employ the multiple files in about expert information in ``expert_statistics\expert_information`` and merge them into a mask file.
```bash
python pruning/expert_selection_mix_domain.py \
    --expert_info_dir expert_statistics/expert_information \
    --expert_mask expert_statistics/expert_mask/mixed_domain_128.json \
    --target_number 128
```

## 4. Model Pruning
---

We provide two modes of model pruning using the [sglang](https://github.com/InternLM/sglang) inference framework:

- **Quick evaluation**: Applies gating masks without changing model weights. The full model is loaded, but only selected experts are activated at inference.
- **Actual pruning**: Removes unused expert weights based on the gating mask, reducing the model size.

### 4.1 Quick Start

> [!NOTE]  
> In the following steps, the official DeepSeek-R1 model weights from Hugging Face should be used

1. Replace the code in:
```
path-to-your-conda/envs/easyep/lib/python3.10/site-packages/sglang
```
with the contents of:
```
sglang/sglang_full/sglang
```

2. Modify the mask file path in `sglang/srt/models/deepseek_v2.py` at line 68:
```python
current_fp = "EASYEP/expert_statistics/expert_mask/aime23_full_br_128.json"  # Replace with your own mask path
```

3. Launch the sglang inference server:
```bash
GLOO_SOCKET_IFNAME=bond0 NCCL_SOCKET_IFNAME=bond0 \
python3 -m sglang.launch_server \
    --model-path path/to/DeepSeek-R1 \
    --tp 8 --dist-init-addr localhost:5002 \
    --trust-remote-code \
    --mem-fraction-static 0.9 \
    --host 0.0.0.0 --port 60000 \
    --context-length 32768 \
    --max-prefill-token 32500 \
    --disable-cuda-graph 
```

> ⚠️ In this mode, all 256 routed experts are still loaded for compatibility. The masking is applied during the gating stage. See 4.2 for full pruning.

---

### 4.2 Actual Pruning

#### Prune the Model

Run the pruning script with your custom expert mask to remove unused expert weights (using the original HuggingFace version of DeepSeek-R1):

```bash
python pruning/model_prune.py \
    --mask_json expert_statistics/expert_mask/aime23_full_br_128.json \
    --input_dir path/to/DeepSeek-R1 \
    --output_dir pruned_model
```

The script validates the checkpoint layout and updates `config.json` and index
metadata automatically. It deliberately refuses DeepSeek-V4 checkpoints with
hash-routed layers because deleting experts without remapping the token-to-expert
table would corrupt routing.

For DeepSeek-V4-Flash, preserve the first three hash-MoE layers and physically
prune only dynamic layers 3–42 with the dedicated, resumable entrypoint:

```bash
export FULL_MODEL_PATH=/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash
export V4_INFERENCE_DIR=/path/to/DeepSeek-V4-Flash/inference
export CONVERTED_CKPT_PATH=/mnt/docker_data/v4-converted
bash scripts/prepare_v4_pruned_checkpoints.sh
```

If the V4 statistics are absent, this entrypoint first collects them on physical
GPUs 4–7 from local files. It does not download dependencies/models or convert
weights. Existing statistics, masks, and completed checkpoint shards are reused.

For the current four-H100 server, the local conversion, statistics, pruning,
reload validation, and evaluation sequence is also available as one fail-fast
command. It stores MP=4 and both pruned checkpoints below
`/mnt/docker_data/v4-converted` and never downloads or installs. That path is
canonical: a stale `CONVERTED_CKPT_PATH=/mnt/docker_data/v4-converte` override
is ignored. A sole complete checkpoint in the typo directory is adopted by a
same-filesystem rename, not copied:

```bash
bash scripts/run_v4_full_reproduction_gpus_4_7.sh
```

Before every V4 model launch, a read-only gate requires physical GPUs 4–7 to
be exclusive (at most 2048 MiB of pre-existing memory each). Busy processes
are reported by PID and the run stops before conversion or model loading; the
script never kills an unknown shared-server workload. SGLang services started
by the repository run in their own process group and are stopped as a group so
TP workers do not remain after the test.

If the official V4 statistics collector reports that
`fast_hadamard_transform` is missing, use the repair-and-resume wrapper. It
first runs a real CUDA Hadamard operation, skips installation when that passes,
and otherwise installs only `fast-hadamard-transform` into `/opt/sglang-v4`.
For CUDA 13 it installs offline from the complete build-required upstream
v1.1.0 source stored under `third_party/fast-hadamard-transform`, bypassing both
the incorrect `cu122` wheel lookup and the incomplete PyPI source distribution.
The vendored compatibility patch avoids the unused cuSPARSE development-header
dependency and compiles only the H100 `sm_90` target. It never reinstalls CUDA,
PyTorch, or SGLang, then resumes the command above and reuses complete MP=4
shards:

```bash
bash scripts/repair_and_resume_v4_full_reproduction.sh
```

Unless its output paths are overridden, the manual
`prepare_v4_pruned_checkpoints.sh` command above writes the 192- and 128-expert
dynamic-layer checkpoints under `models/`. It keeps every 256-row router
weight/bias and hash `tid2eid` tensor unchanged, and
physically removes only unretained expert weights. The SGLang 0.5.16 patch
applies each EASY-EP mask before TopK and maps selected router IDs onto the
compact expert weights. H100 load/generate/reload remains a required test.

```bash
bash scripts/validate_v4_pruned_checkpoints_gpus_4_7.sh
```

---

#### Deploy the Pruned Model

To evaluate the pruned model using sglang:

1. Replace:
```
path-to-your-conda/envs/easyep/lib/python3.10/site-packages/sglang
```
with:
```
sglang/sglang_pruned/sglang
```

2. If you're using a different version of sglang, make sure to update the following files accordingly:
   - `sglang/srt/models/deepseek_v2.py`
   - `sglang/srt/layers/moe/topk.py`

---

## 5. Evaluation
---

We provide evaluation scripts for math-related tasks. Simply run the script below:

```bash
bash evaluation/scripts/run_eval.sh
```

### V4 Agent-OS, GPQA, and KuveCodeBench

The V4 HTTP client also adapts the local Arrow artifacts for three non-MATH
benchmarks. `kuvecodebench` is the repository name for the bundled
LiveCodeBench-v3 data. Each sample is logged in the same format as the AIME
client, including `correct`, `latency`, `completion_tokens`, and `token_ps`:

```bash
python3 evaluation/run_v4_benchmarks.py \
  --data-name gpqa \
  --target-path results/v4_benchmarks/full/evaluation \
  --base-url http://127.0.0.1:60000/v1 \
  --model /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash \
  --max-tokens 32768 --workers 1 --repeats 1 \
  --temperature 1.0 --top-p 1.0 --thinking
```

Run one request per dataset on all three checkpoints before a full matrix:

```bash
python3 scripts/smoke_v4_benchmark_matrix.py
```

The complete full/prune25/prune50 matrix uses the same TP=4 Marlin and decode
CUDA-graph settings as the V4 high-speed server:

```bash
bash scripts/run_v4_benchmark_matrix.sh
```

GPQA is scored by answer letter. Agent-OS is scored by next-action matching
against the offline trajectory. The bundled LiveCodeBench artifact does not
contain hidden judge tests, so the adapter uses public examples: AtCoder rows
run generated Python against sample stdin/stdout, while LeetCode rows parse
the `Solution` starter signature and invoke the method for each example.
Only rows with unparseable or missing public examples are logged as
`correct=unknown` rather than being counted as failures.

## 6. Citation
```
@article{EASY-EP,
  author = {Zican Dong and Han Peng and Peiyu Liu and Wayne Xin Zhao and Dong Wu and Feng Xiao and Zhifeng Wang},
  title = {Domain Specific Pruning of Large Mixture-of-Experts Models with Few-shot Demonstrations},
  journal = {arXiv preprint arXiv: 2504.06792},
  year = {2025}
}
```
## 7. sglang
/opt/sglang-v4/bin/python -m sglang.launch_server \
  --trust-remote-code \
  --model-path /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash \
  --tp 4 \
  --moe-runner-backend marlin \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  --host 127.0.0.1 \
  --port 60000 \
  --context-length 65536 \
  --mem-fraction-static 0.80 \
  --chunked-prefill-size 8192 \
  --max-running-requests 32 \
  --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode 32 \
  --watchdog-timeout 1800 \
  --decode-log-interval 10 \
  --enable-metrics

## 8. Start V4 Benchmark Evaluation

Run these commands from the repository root. The scripts reserve physical GPUs
4-7, start one TP=4 SGLang service at a time with the high-speed Marlin and
decode CUDA-graph profile, and stop that service before moving to the next
checkpoint. Ensure that port `60000` is not already serving SGLang.

First verify that the full, 25%-pruned, and 50%-pruned checkpoints can all
start and answer one sample from each dataset:

```bash
python3 scripts/smoke_v4_benchmark_matrix.py
```

Then start the full `full x prune25 x prune50` evaluation matrix in the
background. It evaluates `agent_os`, `gpqa`, and `kuvecodebench` with
`max_tokens=32768`, `repeats=1`, and `workers=1`. Set a stable `RUN_ID` so an
interrupted run can be resumed with the same command.

```bash
mkdir -p logs
RUN_ID="v4flash_agentos_gpqa_kuvecodebench_$(date +%Y%m%d_%H%M%S)"
nohup env RUN_ID="${RUN_ID}" bash scripts/run_v4_benchmark_matrix.sh \
  >"logs/${RUN_ID}.log" 2>&1 &
```

The matrix log reports stage transitions. Follow the per-question output from
all variants and datasets with this command (keep the path on one line so shell
brace expansion produces nine log files):

```bash
tail --retry -n 0 -F results/easyep_reproduction/${RUN_ID}/{full,prune25,prune50}/evaluation/{agent_os,gpqa,kuvecodebench}.log \
  | grep --line-buffered '^question='
```

Each line has the form below. `token_ps` is completion tokens divided by the
request latency; `correct=unknown` is used only for KuveCodeBench rows without
runnable public examples and is excluded from the accuracy denominator.

```text
question=1/30 repeat=1/1 correct=yes latency=11.89s completion_tokens=1324 token_ps=111.33 progress=1/30 failures=0
```

To resume after an interruption, make sure the old SGLang service has stopped,
set `RUN_ID` to the original value, and repeat the `nohup` command. Existing
`.partial.jsonl` records are reused and completed questions are skipped.
