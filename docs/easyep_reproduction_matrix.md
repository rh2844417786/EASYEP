# EASY-EP 三组完整评测矩阵

## 实验协议

仓库中有正式数学评分器的测试集共有三个：

| 数据集 | 题数 | 用途 |
|---|---:|---|
| AIME24 | 30 | 测试 |
| HMMT Feb 2025 | 30 | 测试 |
| AIME25 | 30 | 测试 |

`dataset/aime23_full` 的 29 条记录用于 calibration；默认从采集结果中按原
EASY-EP 协议选择 25 条样本生成 mask，不能把它混入上述测试集。其他 Arrow
数据暂未接入仓库评分器，因此本矩阵不虚构 GPQA、代码或 Agent 指标。

V4 前 3 个 hash-MoE 层不剪，后 40 个动态 MoE 层按 EASY-EP mask 剪枝：

| 变体 | 动态层剪枝率 | hash 层 0–2 | 动态层 3–42 | 43 层 expert slots 实际减少 |
|---|---:|---:|---:|---:|
| full | 0% | 256 | 256 | 0% |
| prune25 | 25% | 256 | 192 | 23.2558% |
| prune50 | 50% | 256 | 128 | 46.5116% |

MTP 层保持原样，不计入上述 43 个主干层的 slot 比例。报告中的 `25%/50%`
始终指后 40 个动态层的目标剪枝率，不宣称整个 checkpoint 字节数同比减少。

## 先生成 25%/50% mask

可以单独完成 V4 calibration/probe：

```bash
cd /home/jovyan/wangtonghan/EASYEP
export V4_INFERENCE_DIR=/path/to/DeepSeek-V4-Flash/inference
export CONVERTED_CKPT_PATH=/mnt/docker_data/v4-converted
bash scripts/collect_v4_easyep_statistics_gpus_4_7.sh
```

采集脚本只使用本地模型、官方 inference 代码、已有 MP=4 转换权重和仓库内
AIME23 数据，并限制物理 GPU 4–7；它不会下载、安装或转换权重。随后物化入口
会自动生成两个 mask、共享 score tensor 和带 SHA-256/运行时间的 manifest。
它会把前三行强制恢复为 256 个 1，仅对第 3–42 层保留 192/128 个专家。
mask 不是物理 checkpoint；下一步必须执行物理裁剪。

## 生成两个物理 checkpoint

```bash
cd /home/jovyan/wangtonghan/EASYEP
export FULL_MODEL_PATH=/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash
export V4_INFERENCE_DIR=/path/to/DeepSeek-V4-Flash/inference
export CONVERTED_CKPT_PATH=/mnt/docker_data/v4-converted
bash scripts/prepare_v4_pruned_checkpoints.sh
```

该入口先对两个计划做只读 dry-run 和磁盘空间估算，再为 SGLang 0.5.16 应用
版本/源码锚点校验的 mask-routing 补丁，最后逐 shard 写入并验证两个 checkpoint。
若统计尚不存在，它会先自动调用上述 GPU 4–7 采集脚本；统计、mask 或已完成
shard 存在时直接复用，不重复计算。
中断后重复同一命令会复用已完成且 header key 完全匹配的 shard。它不会执行
`apt`、`pip`、`uv`、`curl`、`wget` 或模型下载。

生成结束后先做结构检查以及两个产物各两次 load/generate 验收：

```bash
bash scripts/validate_v4_pruned_checkpoints_gpus_4_7.sh
```

第二次独立启动用于验证 reload；任一 checkpoint 未通过时不要启动完整矩阵。

## 运行三组 checkpoint

把经过独立 load/generate/reload 验证的两个剪枝 checkpoint 放入仓库的
`models/` 目录，使用以下固定布局：

```text
models/
├── v4-prune25-keep192/
│   ├── config.json
│   ├── model.safetensors.index.json
│   └── *.safetensors
└── v4-prune50-keep128/
    ├── config.json
    ├── model.safetensors.index.json
    └── *.safetensors
```

然后运行：

```bash
cd /home/jovyan/wangtonghan/EASYEP

export FULL_MODEL_PATH=/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash

REPEATS=5 WORKERS=1 MAX_TOKENS=32768 \
  bash scripts/run_easyep_reproduction.sh
```

脚本默认读取上述两个仓库内路径；只有需要把 checkpoint 放在其他磁盘时，才应
覆盖 `PRUNE25_MODEL_PATH` 和 `PRUNE50_MODEL_PATH`。

默认依次运行三个变体，使用 GPU 4–7、TP=4、相同采样参数和相同数据顺序。
一次完整实验包含 `90 × 5 × 3 = 1350` 次长推理，可能运行数小时到数天。
中断后设置同一个 `RUN_ID` 重跑即可复用逐题中间结果和已完成的数据集。
续跑会校验 Git commit、三个 checkpoint 指纹和全部科学参数；任何一项变化时会
拒绝把新旧结果写进同一个目录，此时应使用新的 `RUN_ID`。

矩阵会核对 `easyep_expert_mask_by_layer`、物理 expert ID 和
`easyep_pruning` provenance，拒绝前三个 hash 层少于 256 个专家、动态层数量
不统一、router 字段缺失或只是改名/复制的 checkpoint。router weight/bias 始终
保持 256 行；运行时按 mask 选列并映射到紧凑专家 ID。服务端固定关闭
shared-expert fusion、EPLB、redundant experts、CUDA graph 和 MoE A2A backend，
使用普通 TP=4。

## 保存内容

每次运行保存在 `results/easyep_reproduction/<RUN_ID>/`：

- `manifest.json`：Git commit、统一实验配置、checkpoint 指纹和大小；
- `<variant>/server.log` 与 `smoke.log`；
- `<variant>/evaluation/*.jsonl`：逐题输出、正确性、token usage 和请求时延；
- `<variant>/evaluation/*.log`：进度与错误；
- `<variant>/gpu_telemetry.csv`：每 2 秒逐卡显存、利用率和功耗；
- `<variant>/variant_result.json`：启动、smoke、评测和总运行时间；
- `summary.json`、`summary.csv`、`REPORT.md`：三组数值及相对 full 的准确率差。

报告包含准确率、P50/P95/P99 请求时延、wall-clock、checkpoint 字节数和
NVIDIA-SMI 显存峰值。它不等价于 TTFT、TPOT、P99 goodput，也不能单凭
TP=4 结果复现论文的 8 卡吞吐结论。
