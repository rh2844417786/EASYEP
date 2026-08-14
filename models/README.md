# Local pruned checkpoints

Store materialized pruning outputs here. The reproduction wrapper uses these
two directories by default:

```text
models/v4-prune25-keep192/  # 192 of 256 routed experts
models/v4-prune50-keep128/  # 128 of 256 routed experts
```

Each directory must contain `config.json`, `model.safetensors.index.json`, and
every safetensors shard referenced by that index. Checkpoint contents are
ignored by Git intentionally: keep them on the experiment server or another
large-volume storage system instead of uploading them to GitHub.

DeepSeek-V4 hash-routed checkpoints still require a validated token-to-expert
remap/runtime. Merely copying or renaming the full checkpoint is not pruning.
