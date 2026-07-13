# Memory2 事件簇召回评测

该评测直接预置经过脱敏的历史事件记忆，只检查检索能否覆盖多个相关事件簇。`cluster_id` 仅用于评分，不会提供给检索系统。

开发集运行：

```powershell
python -m eval.memory2_cluster.run `
  --config config.toml `
  --timelines eval/memory2_cluster/datasets/timelines.jsonl `
  --dataset eval/memory2_cluster/datasets/dev.jsonl `
  --workers 2 `
  --langsmith `
  --experiment-prefix memory2-cluster-dev-v1
```

评测指标包括核心簇召回率、加权簇覆盖率、Cluster MRR、无关簇比例、禁止簇命中率、重复簇比例和上下文预算效率。`test.jsonl` 是冻结候选，不应在根据开发集调整检索策略前运行。
