# Agent Interviewer RED Baseline

Date: 2026-07-06

## Evidence source

The repository's existing [`面试-6-23.md`](../../../面试-6-23.md) is a useful pre-Skill artifact. Its repeated structure is:

```text
面试官问题
你的回答
优化后的参考回答
```

This structure appears from the opening introduction onward (for example, lines 5, 13, and 17) and repeats for later technical topics.

## Observed failures against the new requirements

1. **Feedback arrives after each question.** `优化后的参考回答` immediately follows the candidate answer, so the simulation switches into teaching mode before the interview ends.
2. **The route is predetermined.** Sections are an authored sequence of topics rather than a next question selected from the candidate's previous answer.
3. **No hidden evidence ledger is preserved until closing.** The artifact improves answers locally but does not accumulate claims, supporting evidence, contradictions, coverage, and confidence for a final hiring assessment.
4. **Final evaluation is incomplete for the target behavior.** It contains retrospective strengths and wording cautions, but not a calibrated probability range, role fit, explicit uncovered areas, or claim-level credibility assessment.

## Testing limitation

The active collaboration policy does not permit creating subagents unless the user explicitly requests them. This baseline therefore uses an existing artifact as observable RED evidence instead of pretending that a self-authored response is an independent agent run. Automated tests below begin in RED because the required Skill files do not yet exist.

