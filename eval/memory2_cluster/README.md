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

## 从本地消息库抽取候选时间线

`extract_candidates` 使用 SQLite 只读模式，按照本地 windows JSON 中预先确定的互不重叠时间窗口，完整保留窗口内的用户消息、助手回复和主动推送。它只生成尚未标注的候选时间线，不会自动生成 query 或 oracle。

```powershell
python -m eval.memory2_cluster.extract_candidates `
  --db .akashic-workspace/sessions.db `
  --windows .akashic-workspace/eval_candidates/candidate_windows.json `
  --replacements .akashic-workspace/eval_candidates/entity_replacements.json `
  --output .akashic-workspace/eval_candidates/natural_candidate_timelines.jsonl
```

凭据、长数字账号、账号句柄、URL、邮箱和本地路径使用通用规则脱敏；个人经历中的机构名称等语义实体通过本地 replacements JSON 处理。windows、replacements 和抽取后的原始候选均应保留在被 Git 忽略的工作空间内。人工复核并进一步抽象后的合成记忆、cluster oracle 和 query 才能进入公开数据集。

候选时间线冻结后，可以生成仅供人工审核的记忆和事件簇草稿：

```powershell
python -m eval.memory2_cluster.draft_clusters `
  --config config.toml `
  --input .akashic-workspace/eval_candidates/natural_candidate_timelines.jsonl `
  --output .akashic-workspace/eval_candidates/memory_cluster_drafts.jsonl `
  --review-output .akashic-workspace/eval_candidates/memory_cluster_review.md `
  --workers 2
```

生成器不会创建 query 或 oracle。每条记忆必须通过真实 `source_ref` 校验，memory 与 cluster 必须双向一致；失败任务会记录错误，同时保留已经成功的结果，重新运行时支持断点续跑。审核表会优先列出低置信度和 assistant-only 记忆，人工确认后才能进入下一阶段。
