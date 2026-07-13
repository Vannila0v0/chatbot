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

## 热度公式配对 A/B

该实验专门比较两条排序链路，除热度参数外保持候选记忆、query embedding、关键词结果和 RRF 逻辑一致：

- Baseline：`hotness_alpha=0`，即语义相似度与关键词排名做 RRF。
- Treatment：`hotness_alpha=0.2`，先将 reinforcement 与时间衰减形成的热度和语义分数混合，再与同一份关键词排名做 RRF。

数据分开报告，不能把两者混成一个平均分：

- `natural_dev.jsonl`：先冻结整段脱敏日常时间线，再从时间线中派生 query，用于观察自然分布表现。
- `challenge_dev.jsonl`：诊断集，其中 benefit 案例验证公式应当生效的场景，guardrail 案例检查旧但重要事实被错误遗忘、高频噪音压过稳定事实等副作用。
- `natural_test.jsonl`：冻结测试候选，在参数和规则定稿前不要运行。

运行自然开发集：

```powershell
python -m eval.memory2_cluster.compare `
  --config config.toml `
  --timelines eval/memory2_cluster/datasets/natural_timelines.jsonl `
  --dataset eval/memory2_cluster/datasets/natural_dev.jsonl `
  --workers 2 `
  --langsmith `
  --experiment-prefix memory2-hotness-natural-dev-v1
```

运行挑战开发集：

```powershell
python -m eval.memory2_cluster.compare `
  --config config.toml `
  --timelines eval/memory2_cluster/datasets/challenge_timelines.jsonl `
  --dataset eval/memory2_cluster/datasets/challenge_dev.jsonl `
  --workers 2 `
  --langsmith `
  --experiment-prefix memory2-hotness-challenge-dev-v1
```

报告同时给出 weighted cluster coverage、core recall、MRR、nDCG@K、偏好簇 pairwise accuracy、forbidden rate 和 irrelevant rate。单案例的“改善/退化”使用方向一致的 Pareto 判定：所有变化中只有改善则记为改善，只有变差则记为退化，同时存在好坏变化则单列为 mixed，避免用任意加权总分掩盖风险。
