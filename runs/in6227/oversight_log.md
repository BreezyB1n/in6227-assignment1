# Oversight log - in6227

- Invocation: `/tabular-classification-report dataset.zip --name in6227`
- Date: 2026-10-11; model: `claude-opus-5-5`; interface: Claude Code 2.1.278 (desktop app, agent harness)
- Human policy: the human set the design constraints of the skill up front (three
  question rounds, below), delegated both checkpoints to independent reviewer sub-agents,
  and asked for every review to be logged here. The human accepts the final report.

## [design] Human decisions before the skill existed
- Actor: human (answering the agent's questions)
- Decisions:
  1. Variant 2 (reusable skill) rather than a single hand-made comparison.
  2. Hybrid skill: deterministic scripts compute every number; the LLM only decides and argues. Report numbers must be traceable to JSON.
  3. Two checkpoints (after the plan, before the final report); review may be delegated to a sub-agent but must be logged.
  4. Exactly two contrasting models (interpretable linear + tree ensemble) plus a prior baseline.
  5. Stratified 5-fold CV on train only; test used once; PR-AUC / macro-F1 as primary metrics; threshold tuned on out-of-fold predictions; McNemar test.
  6. Generic profiler detectors (sentinel zeros, outliers, derived features, leakage, rare levels, missing labels).
  7. Generalisation evidence on two other datasets (breast_cancer: numeric binary; OpenML car: categorical multiclass).
  8. English report in LaTeX; identity fields left as placeholders.

## [stage 1] Profile
- Actor: script (`profile_data.py`)
- Finding: 31,109 / 13,333 rows after dropping 3 + 1 rows with missing labels; 24 % positive; 24 flags.
  Notable: `composite_rank = 0.006*aptitude_score + 0.004*performance_score` (R^2 = 0.9999999);
  `performance_score` and `stability_index` are zero together in 265 rows, all labelled `no`;
  `load_ratio`/`activity_duration` zero together in 38 rows; 17 rare `region` levels.
- Response: all 24 flags must be addressed in plan.json.

## [checkpoint 1] Plan review by reviewer sub-agent (fresh context, read-only)
- Actor: reviewer-agent (general-purpose sub-agent; inputs: profile.md, profile.json, plan.json, decision-rules.md, run_pipeline.py)
- Verdict: no blockers, 2 major, 10 minor. Full points and responses:
  1. (minor) `high_correlation` flag not addressed -> **accepted**: added `high_correlation: keep` with interpretation caveat.
  2. (minor) clip `why` misquoted the outlier range (index_weight is 0.52 %) -> **accepted**: corrected per-column figures.
  3. (minor) "well-calibrated" LR is only true if class_weight=None wins -> **accepted**: reworded.
  4. (minor) "not recorded" reading of the all-'no' zero block is weakly argued; it looks structural -> **accepted**: rationale rewritten as "structurally different group"; action unchanged because NaN + indicator keeps the signal.
  5. (minor) HGB min_samples_leaf capped at 200 vs house rule n/50 ~ 620 -> **accepted**: range widened to [10, 600]. (Note: a pre-review smoke run had also put the best value at 198, i.e. at the old boundary - consistent with this point.)
  6. (minor) all-'no' zero block could be post-outcome information -> **accepted as caveat** in the report; no single feature reaches the leakage threshold (max single-feature AUC 0.779).
  7. (minor) OOF threshold uses the same folds that chose the hyper-parameters (slightly optimistic) -> **accepted as stated limitation**; nested CV not run (5x cost for a threshold choice).
  8. (minor) the two models also differ in preprocessing, so the gap is not pure capacity -> **accepted**: will be stated; ablations A4/A5 quantify the composite_rank part.
  9. (major) zero_as_missing applies to both models but was only ablated for LR -> **accepted**: added A2_hgb_keep_zeros.
  10. (major) ablation reference is the search-best score (selection-biased) and variants reuse hyper-parameters tuned for the chosen configuration -> **partly accepted**: script now reports the sd of paired per-fold differences and a `within_noise` flag; the report treats |delta| < sd as "no detectable effect" and states that deltas are conservative for the variants. Not re-tuning each variant (would multiply cost by 6 for small expected effects).
  11. (minor) rare-level pooling and missing indicators asserted, not tested -> **rejected**: region has single-feature AUC 0.517 and missingness is <= 7 cells per column, so the possible effect is below CV noise; page budget better spent elsewhere.
  12. (minor) no test separating interactions from non-linear numeric effects (e.g. LR + splines) -> **rejected for this run** (estimator not in the skill's registry); recorded as future work in the report.
- Disclosure: while the scripts were being developed, one smoke run of an earlier plan printed test metrics
  (LR AP 0.700, HGB AP 0.706). No plan change above was motivated by them; every change traces to a reviewer point
  or to CV-only evidence (the min_samples_leaf boundary).

## [stage 3] Run and diagnostics
- Actor: script, then agent
- Finding: HGB best `min_samples_leaf` = 30 was flagged as "at search boundary" of [10, 600]. On inspection this was a
  false positive of the boundary detector (linear-scale 5 % rule on a 60x range). Fixed the detector to use a log scale
  for wide positive ranges (`run_pipeline.py::at_boundary`) and re-ran; no hyper-parameter range was changed.
- Finding: widening `min_samples_leaf` to [10, 600] (checkpoint-1 point 5) changed the random draws; HGB CV AP is
  0.7207 vs 0.7217 in the pre-review smoke run. The difference is far below the fold sd (0.015), i.e. the search
  surface is flat - randomised-search noise, not an effect of the range.
- Checks: LR converged in 213/5000 lbfgs iterations; HGB early-stopped at 198/2000 rounds; CV train-minus-validation
  AP gap 0.003 (LR) and 0.037 (HGB), both below the 0.05 alarm level.
- Ablation outcomes vs plan rationale:
  * zero_as_missing (A1, A2): no detectable effect for either model (|delta| < fold sd). Decision kept (harmless,
    makes the structural group explicit) but the report must say it did not matter.
  * HGB keeps composite_rank (A5): **not supported** - dropping it changes AP by +0.0002 (within noise), although
    permutation importance ranks it 2nd. Report as a finding: importance is shared with aptitude/performance.
  * LR clipping (A3): -0.0006 without clipping, outside fold noise but p = 0.09: weak support.
  * Native categorical (A6): one-hot costs 0.0035 AP, p = 0.005: supported.

## [stage 5] Render iterations (CV-independent, presentation only)
- Actor: agent
- First render showed the figure floated to page 3 while the page check passed: the check read the page of a
  `\label` placed before pending floats. **Skill bug fixed**: the main report is now compiled on its own and its
  page count read from `\@abspage@last`; a `\FloatBarrier` closes the main part. Title escaping and legend overlap fixed.
- After adding the reviewer-requested content the report reached 3 pages; cut prose, dropped a redundant table
  column, reduced figure height, allowed tables at page bottom. Final: 2 pages.

## [checkpoint 2] Report review by reviewer sub-agent (fresh context, read-only, may run read-only Python)
- Actor: reviewer-agent. Inputs: report.tex, numbers_audit.md, results.json, profile.json, plan.json.
- Numbers: all 68 tokens resolve; literals (0.009, 24, 21, 10 %, 30, 15/7/8) verified; reviewer re-derived
  HGB tuned precision/recall/F1/balanced accuracy (`results.models.hgb.test.at_tuned`), LR CV AP mean/sd
  (`results.models.lr.cv.average_precision`), baseline macro-F1 and every ablation sd. No mismatch.
- Points and responses:
  1. (major) "both models now miss under a quarter of positives" is false for HGB (793/3168 = 25.03 %) -> **accepted**: sentence replaced by token-based counts (724 and 793 of 3,168). A real factual error caught by review; it was the one claim derived by the agent in prose rather than by a token.
  2. (major) issues table said zero->NaN "removes an artificial point mass (tested: A1, A2)" although both ablations show no effect (slightly higher without) -> **accepted**: now "on prior grounds ... ablations show no measurable effect".
  3. (minor) A4 delta -7.6e-6 printed as "+0.0000" -> **accepted**: values below 5e-5 print as unsigned 0.0000.
  4. (minor) "neither model over-fits" too strong (HGB gap 11x LR's) -> **accepted**: "over-fitting is limited (larger for HGB)".
  5. (minor) "signal is mostly additive" should be hedged -> **accepted**: "appears largely additive", with the preprocessing caveat.
  6. (minor) A3 "cost" implies an effect with p = 0.093 -> **accepted**: "changed AP by ... weak evidence".
  7. (minor) search ranges not shown -> **accepted**: config table now lists ranges (renderer change, applies to all datasets).
  8. (minor) AP rationale only implicit -> **accepted**: one clause added.
  9. (minor) 1 % pooling rationale missing -> **accepted**: one clause added.
  10. (minor) model version not a separate field -> **accepted**: "Claude Opus 5.5 (claude-opus-5-5)".
  11. (minor) no calibration / runtime discussion -> **rejected**: page budget; neither is part of the comparison question (search time is in results.json).
  12. (minor) "threshold matters more than model" uses an arbitrary F1 objective -> **no change**: already stated in Limitations.
- Independent check by the agent (outside the pipeline, plain pandas on the raw CSVs): 31,109 / 13,333 labelled rows;
  positive share 0.2399; perf/stability zero block = 265 rows, all `no`; composite_rank residual <= 0.0005;
  HGB recall from the confusion matrix 2375/3168 = 0.7497 = reported value; HGB miss rate 0.2503 (confirms point 1).
- Note (later): profiler improved during the generalisation runs (high-correlation pairs collapsed into one flag; drift requires KS p < 0.01). Re-profiling in6227 gives 22 flags instead of 24 (the 3 correlation flags became 1); no flag that drove a decision changed. The report cites the count by token.
