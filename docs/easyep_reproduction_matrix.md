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

对 256 个 routed experts，三个变体必须是：

| 变体 | 剪枝率 | 每层保留专家 |
|---|---:|---:|
| full | 0% | 256 |
| prune25 | 25% | 192 |
| prune50 | 50% | 128 |

## 先生成 25%/50% mask

完成 `deepseek_v4_flash_reproduction.md` 的 V4 calibration/probe 后运行：

```bash
cd /home/jovyan/wangtonghan/EASYEP
TOKEN_STATS=expert_statistics/token_information/aime_v4.jsonl \
  bash scripts/prepare_easyep_masks_25_50.sh
```

脚本生成两个 mask、共享 score tensor 和带 SHA-256/运行时间的 manifest。
mask 不是物理剪枝 checkpoint。当前仓库仍不支持 V4 hash-router 的安全物理
剪枝；不得把完整模型配上不同名称重复评测。

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

V4 checkpoint 包含 hash layers 时，矩阵默认拒绝 pruned 变体。这是安全门禁，
不是需要随手关闭的报错。只有在另一个实现中完成并验证 hash token→expert
重映射、checkpoint loader 和 MoE kernel layout 后，才可显式设置：

```bash
ALLOW_HASH_ROUTED_PRUNED_CHECKPOINTS=1 \
  bash scripts/run_easyep_reproduction.sh
```

该开关只表示“用户提供的外部实现已验证”，本仓库不会因此自动获得 V4 剪枝
能力。

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
