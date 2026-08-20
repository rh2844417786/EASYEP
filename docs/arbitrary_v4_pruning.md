# DeepSeek-V4-Flash：一次校准生成任意剪枝比例

## 结论

昂贵的 GPU 校准产物是 `expert_statistics/token_information/aime_v4.jsonl`。
只要模型、Tokenizer、聊天模板、校准数据和评分定义不变，这个文件可以反复用于
不同的动态层专家保留数量，不需要再次启动完整模型做校准前向。

专家数量是离散的，因此“任意比例”最终必须映射为整数 `K`。本实现支持：

- `TARGET_EXPERTS`：直接指定每个动态 MoE 层保留的专家数量；
- `PRUNE_PERCENTAGES`：指定希望删除的百分比；
- 两者均可使用逗号分隔，一次生成多个目标；
- layer 0–2 始终保留全部 256 个 hash-routing experts；
- layer 3–42 使用统一的 `K`；
- `K` 必须不小于模型的 per-token router Top-K，且不大于 256。

## 只生成掩码

```bash
cd /home/jovyan/wangtonghan/EASYEP

TARGET_EXPERTS=224,192,160,128 \
PRUNE_PERCENTAGES=10,30,50 \
  bash scripts/prepare_easyep_masks.sh
```

该命令只读取已有 token statistics，在 CPU 上聚合一次 `43 x 256` 专家分数，
然后为所有目标分别执行逐层 Top-K。它不会加载模型、使用 GPU、下载文件或写入
checkpoint。

默认输出位于：

```text
expert_statistics/expert_mask/aime_v4_scores.pt
expert_statistics/expert_mask/aime_v4_mask_manifest.json
expert_statistics/expert_mask/aime_v4_prune<实际比例>-keep<K>.json
```

例如请求 `PRUNE_PERCENTAGES=30` 时，256 个专家无法精确删除 30%。实现向上取整
保留数量，得到 `K=180`，实际动态层删除比例为 29.6875%，输出名称为：

```text
aime_v4_prune29p6875_keep180.json
```

该取整策略保证实际删除比例不会超过用户请求值。

## 生成实体剪枝模型

```bash
TARGET_EXPERTS=224,192,160,128 \
MODEL_OUTPUT_ROOT=/mnt/docker_data/v4-converted \
  bash scripts/prepare_v4_pruned_checkpoints_any.sh
```

如果 token statistics 已存在，脚本明确复用它，不会再次校准；如果不存在且
`AUTO_COLLECT_STATS=1`，才会调用一次 GPU collector。所有目标先执行只读 dry-run，
全部通过后才升级 SGLang 补丁并逐个实体化权重。

输出目录遵循：

```text
/mnt/docker_data/v4-converted/v4-prune<实际比例>-keep<K>/
```

实体剪枝仍保留完整 256 行 router weight/correction bias。运行时先按照持久化 mask
选择 router logits，再执行 Top-K，并把原专家编号映射到紧凑物理编号。

## H100 验收

```bash
MASK_MANIFEST=expert_statistics/expert_mask/aime_v4_mask_manifest.json \
MODEL_OUTPUT_ROOT=/mnt/docker_data/v4-converted \
RELOAD_PASSES=2 \
  bash scripts/validate_v4_pruned_checkpoints_any_gpus_4_7.sh
```

验收会对 manifest 中每个目标执行结构校验以及两次独立的
load/generate/stop/reload。静态单元测试通过不等于新的 `K` 已在 H100/Marlin
路径验证；只有该脚本最终输出 `PASS`，才能把该比例标记为可运行。

## 比例口径

指定的百分比仅表示 40 个动态层内部的专家删除比例。前三个 hash 层不剪枝，
因此 43 个主层的专家槽位删除比例为：

```text
40 * (256 - K) / (43 * 256)
```

这仍不等于整模型参数压缩率，因为 Attention、Embedding、Shared Experts、Router
和 MTP 等参数没有按相同比例删除。

## 何时必须重新校准

只有改变 `K` 或剪枝百分比时，不需要重新校准。以下变化需要重新采集统计：

- 更换模型 checkpoint 或 router/expert 权重；
- 更换 Tokenizer、聊天模板或 thinking mode；
- 更换目标领域或校准样本；
- 修改专家评分公式；
- 希望在剪枝后重新估计路由分布并进行迭代剪枝。

保留一个二值 mask 不足以生成其他比例；应保留 token-statistics JSONL，或者至少
保留完整的专家分数矩阵及其来源信息。
