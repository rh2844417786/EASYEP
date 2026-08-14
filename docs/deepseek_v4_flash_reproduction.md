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

### 3.1 先采集当前 Docker 环境

若不确定当前容器是否具备 SGLang、PyTorch/CUDA、启动参数及模型文件依赖，
先运行零第三方依赖的采集器。它只读取白名单信息，并在目标 GPU 上执行一个
64×64 FP16 矩阵乘；不会加载模型权重：

```bash
cd /path/to/easy-ep
export MODEL_PATH=/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash
python3 tools/v4_environment_report.py \
  --gpus 4,5,6,7 \
  --output reports/v4_environment.md
```

即使缺少 SGLang、Torch 或 `nvidia-smi`，采集器也会尽量完成并生成 Markdown，
默认不会因 `FAIL` 检查返回非零退出码。需要在自动化中把缺失依赖视为失败时，
追加 `--strict`；不希望初始化 CUDA 时可追加 `--skip-cuda-smoke`。

把生成的 `reports/v4_environment.md` 完整发回后，再决定使用现有容器、修复
依赖，还是切换固定版本的 SGLang 镜像。报告不会输出完整环境变量、认证状态、
Docker inspect 或模型权重，并会遮盖常见令牌格式。

### 3.2 启动原始 V4 服务

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

如果当前只能使用物理 GPU 4–7，可运行独立的四卡可行性测试。该脚本固定
`CUDA_VISIBLE_DEVICES=4,5,6,7` 和 `TP=4`，自动等待服务、发送 smoke
请求并停止本次启动的服务：

```bash
export MODEL_PATH=/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash
bash scripts/test_v4_flash_gpus_4_7.sh
```

V4 的 mHC 权重加载会通过 DeepGEMM 即时编译 CUDA kernel；因此即使 MoE
backend 使用 Marlin，也不能只安装 PyTorch 自带的 CUDA runtime。当前验证
路径要求 Docker 内已经存在 CUDA 13.x NVCC、PyTorch CUDA 13 构建、SGLang
0.5.16、物理 GPU 4–7 和完整的本地模型权重。运行下面的兼容入口会先做只读
预检，再启动四卡测试：

```bash
cd /path/to/easy-ep
# 可选：只做预检，通过后退出，不启动模型。
bash scripts/validate_v4_flash_runtime.sh

# 预检通过后自动继续四卡 smoke 测试。
bash scripts/repair_and_test_v4_flash_gpus_4_7.sh
```

历史文件名中虽然保留了 `repair`，但该入口现在**不会**执行 `apt`、`curl`、
`wget`、`pip`、`uv` 或模型下载。它先检查系统中是否已有这些传输/安装进程；
若检测到重叠任务会直接停止（确认无关后才可显式设置
`IGNORE_ACTIVE_DOWNLOADS=1`）。随后检查 CUDA 13.x NVCC、SGLang 0.5.16、
PyTorch CUDA 13、四张目标 GPU、SGLang 启动参数以及本地权重索引和全部分片。
任一条件缺失都只报错，不自动修复。启动前还强制设置
`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，避免缺失权重时转为联网下载。
四卡首次 forward 还可能触发耗时的 DeepGEMM JIT。测试入口将 SGLang hard
watchdog 和 smoke HTTP timeout 都设为 1800 秒，避免默认 300 秒 watchdog
在 kernel 即将完成时主动终止服务；可分别用 `WATCHDOG_TIMEOUT` 和
`SMOKE_TIMEOUT` 覆盖。

验证与测试总日志写入 `logs/v4_validate_and_test_*.log`，入口摘要写入
`logs/v4_validate_and_test_*_summary.txt`，SGLang 测试结果仍写入独立的
`logs/v4_gpus_4_7_*_summary.txt`。如确实需要重新安装 CUDA，必须单独、显式地
运行 `scripts/install_v4_cuda_toolchain.sh`；验证入口不会调用它。

这是诊断路径，不是论文的 8 卡基线。若出现 hidden-size mismatch、FP4
权重 shape 错误或 OOM，应保留脚本输出的 `logs/v4_gpus_4_7_*.log`；不能把
四卡失败直接归因于 EASY-EP，也不能把四卡 smoke 成功当作吞吐复现完成。

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
TOKEN_STATS=expert_statistics/token_information/aime_v4.jsonl \
  bash scripts/prepare_easyep_masks_25_50.sh
```

该入口分别生成动态层 keep-192/keep-128 mask，并把前三个 hash 层强制设为
全 1。不要直接使用会把 43 层统一裁成 128 的旧命令。

混合域：

```bash
python pruning/expert_selection_mix_domain.py \
  --expert-info-dir expert_statistics/expert_information \
  --expert-mask expert_statistics/expert_mask/mixed_v4_128.json \
  --target-number 128
```

新实现不再固定 58 层，并修复了混合域脚本中 `w` 未定义、`--target-number` 被忽略、零分母及 shape 不一致等问题。

## 8. Gate 5：保留 hash 层，物理裁剪后 40 层

本仓库采用论文式 mask 路由，不裁剪 router 参数：

- layer 0–2：保留全部 256 个专家、gate weight 和 `tid2eid`；
- layer 3–42：只删除未保留的专家权重，并把保留专家连续重编号到 0–191 或
  0–127；gate weight 和 correction bias 仍完整保留 256 行；
- MTP：保持原样；
- config：全局 `n_routed_experts` 保持 256，写入完整的
  `easyep_expert_mask_by_layer` 和带哈希的 `easyep_pruning` provenance；
- runtime：对 SGLang 0.5.16 做版本/源码锚点校验；动态层先用 mask 从完整
  router logits/correction bias 选择保留列，再执行 TopK，紧凑 ID 与物理专家
  一一对应。运行时关闭 shared-expert fusion、EPLB、redundant experts 和 CUDA
  graph，并限定普通 TP 路径（不启用 MoE A2A backend）。

生成两个 checkpoint：

```bash
cd /home/jovyan/wangtonghan/EASYEP
export FULL_MODEL_PATH=/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash
bash scripts/prepare_v4_pruned_checkpoints.sh
```

入口会在写权重前检查 43×256 的源 expert key、前三行 mask 全 1、后 40 行
分别恰好 192/128、所有 shard header 与 index 一致，并估算磁盘空间。逐 shard
写入使用临时文件和原子替换；中断后可复用已验证 shard。最终 index/config 在
权重完成后才写入，并再次验证 key 集、总 tensor 字节数和计划指纹。

在正式矩阵前，让两个产物各执行两次独立的 load/generate/stop（第二次即 reload
验证）：

```bash
bash scripts/validate_v4_pruned_checkpoints_gpus_4_7.sh
```

该验收固定使用物理 GPU 4–7，关闭 shared-expert fusion，并为每次启动保存完整
日志和自动摘要。只有脚本最终输出 `PASS`，才进入评测矩阵。

恢复未修改的 SGLang 文件：

```bash
/opt/sglang-v4/bin/python \
  scripts/patch_sglang_v4_heterogeneous_experts.py --restore
```

代码和 checkpoint 结构检查不等于 H100 推理通过。必须继续执行下面的 load、
generate、停止后 reload 验收，才能把数值标为 V4 物理剪枝结果。

## 9. 结果验收顺序

1. smoke 请求成功，保存服务版本、完整启动命令和 GPU 拓扑；
2. 三个数据集未剪枝基线完成，保存 JSONL、时延与显存峰值；
3. V4 probe 一条样本 43 层齐全，无 NaN/Inf；
4. 25 条 calibration 生成 mask，检查前三层 256 个 1，后 40 层为 192/128；
5. 两个物理 checkpoint 通过 index、shape、完整 router 字段、mask 和 provenance 验证；
6. 192 模型 load/generate/stop/reload 成功，再对 128 模型执行同样验收；
7. 三组评测矩阵完成后才比较准确率、运行时间、checkpoint bytes 与峰值 HBM；
8. TP=4 结果不外推为论文 TP=8 的 TTFT、TPOT、P99 或 goodput 结论。
