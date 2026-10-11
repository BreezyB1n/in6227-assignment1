# Oversight log - car (generalisation example)
- Invocation: /tabular-classification-report examples/car/car.csv
- Date: 2026-10-11; model: claude-opus-5-5; interface: Claude Code 2.1.278 (desktop app)
- Human policy: generalisation demo. Reviewer checkpoints were **not** run for the examples (cost); the plan follows
  references/decision-rules.md and every deviation is stated in plan.json.

## [stage 1] Profile
- 1 flag (class imbalance, ratio 18.6:1). No numeric columns.
## [stage 2] Plan
- Multiclass -> macro-F1 primary, no threshold; n < 2000 -> 10 folds; HGB min_samples_leaf lowered to [2, 30] (smallest class ~52 rows).
## [stage 3] Run
- HGB max_leaf_nodes = 64 at the upper boundary -> widened once to [8, 128] per stage-3 rule; re-run selected 66 (inside the range).
- Ablation one-hot vs native: +0.007 macro-F1 for one-hot, within fold noise (sd 0.028).
