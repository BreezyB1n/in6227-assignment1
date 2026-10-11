# Oversight log - breast_cancer (generalisation example)
- Invocation: /tabular-classification-report examples/breast_cancer/breast_cancer.csv
- Date: 2026-10-11; model: claude-opus-5-5; interface: Claude Code 2.1.278 (desktop app)
- Human policy: generalisation demo. Reviewer checkpoints were **not** run for the examples (cost); the plan follows
  references/decision-rules.md and every deviation is stated in plan.json.

## [stage 1] Profile
- Detectors initially produced 66 flags: one per highly correlated pair and KS drift flags on a 114-row hold-out.
  Profiler improved (correlation pairs collapsed into one flag; drift needs KS p < 0.01) -> 18 flags.
## [stage 2] Plan
- Rules applied: minority share 0.37 >= 0.35 -> ROC-AUC primary, balanced-accuracy threshold; n_train 455 < 1000 ->
  Random Forest instead of HGB; n < 2000 -> 10 folds; concavity zero block kept (physically meaningful zeros).
## [stage 3] Run
- RF min_samples_leaf = 1 is at the range boundary, but 1 is the smallest legal value -> not widened.
- LR vs RF: no detectable difference (McNemar 2/2, p = 1.0; corrected t-test p = 0.23).
