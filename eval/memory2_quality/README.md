# Memory2 质量评测

本评测独立检查 Memory2 的写入类型、短期状态误写、冲突更新、实体属性隔离、噪音事实提取和召回质量。每个 case 使用独立 workspace，不会写入真实用户记忆。

```bash
python -m eval.memory2_quality.run \
  --config config.toml \
  --dataset eval/memory2_quality/datasets/smoke.jsonl \
  --mode all \
  --workers 2
```

`--mode write` 只检查写入，`--mode recall` 使用预置标准记忆检查检索，`--mode all` 检查真实写入后的召回。添加 `--langsmith` 后会启用可选 LangSmith 结果同步；未配置 LangSmith 不影响本地报告。
