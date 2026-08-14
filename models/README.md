# Local pruned checkpoints

Store materialized pruning outputs here. The reproduction wrapper uses these
two directories by default:

```text
models/v4-prune25-keep192/  # layers 0..2: 256; layers 3..42: 192
models/v4-prune50-keep128/  # layers 0..2: 256; layers 3..42: 128
```

Each directory must contain `config.json`, `model.safetensors.index.json`, and
every safetensors shard referenced by that index. Checkpoint contents are
ignored by Git intentionally: keep them on the experiment server or another
large-volume storage system instead of uploading them to GitHub.

Generate both directories with `scripts/prepare_v4_pruned_checkpoints.sh`.
The V4 path preserves every hash-routed expert and `tid2eid` table, physically
prunes only expert weights in the following 40 dynamic-MoE layers, and leaves
all 256 router rows intact. Its SGLang 0.5.16 patch applies the saved EASY-EP
mask before TopK and maps selected router IDs to compact expert IDs. Merely
copying or renaming the full checkpoint is not pruning.

Before the full matrix, run
`scripts/validate_v4_pruned_checkpoints_gpus_4_7.sh`; it verifies both layouts
and performs two independent load/generate cycles per checkpoint on GPUs 4–7.
