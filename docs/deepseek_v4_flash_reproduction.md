# DeepSeek-V4-Flash × EASY-EP 复现与诊断

本文档针对 8×H100 节点。它把“原模型能否正确推理”“EASY-EP 统计与选专家”以及“真实删权重”分成三个独立关卡，避免把服务框架问题误判为剪枝结果。

## 1. 已定位的问题

1. DeepSeek-V4-Flash 是 284B 总参数、约 13B 激活参数的混合精度模型，routed experts 为 FP4，其余主要权重为 FP8；不是问题汇报中写的约 138B。
2. `149GB < 4 × 80GB` 只能说明权重字节数看似能放下，不能证明某个 TP/内核组合受支持。CUDA graph、KV/cache、workspace、通信与 FP4 kernel 都需要额外显存和特定布局。
3. 报错时使用的 H100 `TP=4 + Triton fused_moe` 不是 SGLang 当前验证路径。当前官方 H100/原始 FP4 Flash 路径是 `TP=8 + --moe-runner-backend marlin`。
4. 仓库内 `sglang/sglang_full` 与 `sglang/sglang_pruned` 基于 SGLang 0.4.3 和 `deepseek_v2.py`，只适用于论文原始 DeepSeek-V3/R1 路线。不要覆盖到 V4 的当前 SGLang 环境。
5. 原 `pruning/inf_new.py` 内嵌 V3/R1 模型：固定 58 个 MoE 层、sigmoid/softmax router、普通残差，并不理解 V4 的 43 层、3 个 hash-router 层、sqrtsoftplus、mHC 与 FP4 experts。
6. V4 的前三个 hash-router 层包含 token→expert 表。若直接删除/重编号专家而不重写每个 token 的映射，模型语义即损坏。因此当前 `model_prune.py` 会拒绝 V4，而不是静默生成错误 checkpoint。

## 2. 环境必须隔离

| 环境 | 用途 | 依赖 |
|---|---|---|
| `easyep-r1` | 论文原始 R1/V3 复现 | 根目录 `requirements.txt`，含 SGLang 0.4.3 |
| 当前 SGLang 容器 | V4 原模型服务与基线评测 | `lmsysorg/sglang:latest` 或与官方 V4 cookbook 同步的构建 |
| `easyep-v4-probe` | 用 DeepSeek 官方 inference 代码采集内部统计 | `requirements-v4-probe.txt` |
| `easyep-eval` | HTTP 评测与数学判分 | `requirements-eval.txt` |

不要把仓库中的旧 `sglang/` 目录复制进 V4 服务容器。

## 3. Gate 0：先验证原始 V4 服务

服务器上需要一次拿到全部 8 张 H100，而不是只暴露 GPU 4–7：

```bash
cd /path/to/easy-ep
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_PATH=/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash

# 首次只验证 eager forward，绕开 CUDA graph，便于隔离问题。
PROFILE=smoke PORT=60000 bash scripts/serve_v4_flash_h100.sh
```

另开终端发一个最小请求：

```bash
cd /path/to/easy-ep
python3 scripts/smoke_v4_server.py --base-url http://127.0.0.1:60000/v1
```

若成功，停止 smoke 服务并启动官方已验证的吞吐路径：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_PATH=/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash
PROFILE=verified PORT=60000 bash scripts/serve_v4_flash_h100.sh
```

启动脚本会先运行 `tools/v4_preflight.py`，检查模型结构、可见卡数、每卡空闲显存、TP 与 MoE backend。若仍出现 hidden-size mismatch，请保存以下完整信息再定位，不要只截最后一行：

```bash
python -m pip show sglang flashinfer-python torch
nvidia-smi
python3 tools/v4_preflight.py \
  --model-path "$MODEL_PATH" --tp 8 --backend marlin --json
```

## 4. Gate 1：基线评测

评测客户端改为 OpenAI-compatible HTTP，不再导入旧 SGLang DSL；它支持并发上限、失败重试、逐题 checkpoint 与断点续跑。

```bash
python3 -m venv .venv-eval
source .venv-eval/bin/activate
pip install -r requirements-eval.txt

BASE_URL=http://127.0.0.1:60000/v1 \
REPEATS=5 WORKERS=8 MAX_TOKENS=32768 \
bash evaluation/scripts/run_eval.sh
```

V4 官方推荐采样参数为 `temperature=1.0, top_p=1.0`，新脚本默认采用这组参数并打开 Think-High。中途中断后重复同一命令即可从 `*.partial.jsonl` 续跑。先完成未剪枝基线；未通过 Gate 0/1 时，后续分数不能归因于 EASY-EP。

## 5. Gate 2：制作 V4 calibration 数据

仓库自带 Arrow 数据包含原始 `input` 和旧 `input_ids`。为避免 tokenizer/chat encoding 混用，先用 V4 官方 encoder 重新编码：

```bash
python pruning/prepare_v4_calibration.py \
  --source dataset/aime23_full \
  --output dataset/aime23_full_v4 \
  --model-path /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash \
  --inference-dir /path/to/DeepSeek-V4-Flash/inference
```

`--inference-dir` 的父目录还必须有官方 `encoding/encoding_dsv4.py`。

## 6. Gate 3：采集 V4 EASY-EP 统计

`pruning/inf_v4.py` 动态导入官方 `inference/model.py`，不复制或修改上游文件。它采集：

- router top-k expert IDs 与 sqrtsoftplus 权重；
- 未加权专家输出的 L2 norm；
- 在 mHC `hc_post` 后计算的 token counterfactual cosine similarities；
- 43 层（含 3 个 hash-router 层）的真实 layer ID。

先用现有 FP4/MP4 转换权重做一条带同步的诊断，能够把原来的“forward 卡住”定位到具体 layer/stage：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
torchrun --nproc-per-node=4 pruning/inf_v4.py \
  --inference-dir /path/to/DeepSeek-V4-Flash/inference \
  --ckpt-path /mnt/docker_data/v4-converted \
  --config configs/config_v4_flash.json \
  --input-file dataset/aime23_full_v4 \
  --output expert_statistics/token_information/aime_v4_debug.jsonl \
  --limit 1 --trace-first-sample --sync-debug
```

如果日志停在 `moe:start`，优先绕过 FP4 TileLang GEMM：用官方 `convert.py` 将 FP4 数值无损展开到 FP8，并按 8 卡重新分片。该操作不是重新量化训练，但会增加权重体积与计算成本。

```bash
python /path/to/DeepSeek-V4-Flash/inference/convert.py \
  --hf-ckpt-path /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash \
  --save-path /mnt/docker_data/v4-converted-fp8-mp8 \
  --n-experts 256 --model-parallel 8 --expert-dtype fp8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc-per-node=8 pruning/inf_v4.py \
  --inference-dir /path/to/DeepSeek-V4-Flash/inference \
  --ckpt-path /mnt/docker_data/v4-converted-fp8-mp8 \
  --config configs/config_v4_flash_fp8_probe.json \
  --input-file dataset/aime23_full_v4 \
  --output expert_statistics/token_information/aime_v4.jsonl \
  --limit 25 --resume
```

统计模式会为了获得论文定义的未加权 `||E_i(x)||` 再执行一次被激活专家，因此它不是吞吐基准。

## 7. Gate 4：生成 expert mask

```bash
python pruning/expert_selection.py \
  --input-file expert_statistics/token_information/aime_v4.jsonl \
  --output-file expert_statistics/expert_information/aime_v4.pt \
  --expert-mask expert_statistics/expert_mask/aime_v4_128.json \
  --num-experts 256 --target-number 128 --num-samples 25
```

混合域：

```bash
python pruning/expert_selection_mix_domain.py \
  --expert-info-dir expert_statistics/expert_information \
  --expert-mask expert_statistics/expert_mask/mixed_v4_128.json \
  --target-number 128
```

新实现不再固定 58 层，并修复了混合域脚本中 `w` 未定义、`--target-number` 被忽略、零分母及 shape 不一致等问题。

## 8. 尚未完成的 V4 关卡

生成 V4 mask 还不等于完成论文复现。以下两项需要单独实现并做一致性测试：

1. **当前 SGLang 的 V4 quick-mask 路由**：动态 router 层可以在 top-k 前屏蔽专家；前三个 hash 层不能简单套用同一操作。旧 `sglang_full` 补丁不可复用。
2. **真实物理剪枝**：需要支持“hash 层保留 256、其余层保留 128”的异构专家数，或提出并验证 token→expert 重映射策略；同时修改 checkpoint loader、MoE kernel layout 和配置语义。

在这两项完成前，可以报告“V4 原模型基线”和“V4 calibration mask/稳定性”，但不能报告“V4 EASY-EP 已完成 2× 参数压缩或部署吞吐提升”。根目录 `model_prune.py` 仍可用于无 hash-router 的 V3/R1 checkpoint，并会自动更新 `n_routed_experts` 与 index metadata。

## 9. 结果验收顺序

1. smoke 请求成功，保存服务版本、完整启动命令和 GPU 拓扑；
2. 三个数据集未剪枝基线完成，保存 JSONL、时延与显存峰值；
3. V4 probe 一条样本 43 层齐全，无 NaN/Inf；
4. 25 条 calibration 生成 mask，检查每层恰好 128 个 1；
5. quick-mask 实现后先比较“mask 全 1”与原服务逐 token logits/生成一致性；
6. 再评测 mask=128；
7. 只有真实删权重模型成功 load、generate、reload 且输出可复现后，才测 TTFT、TPOT、吞吐、P99 与峰值 HBM。
