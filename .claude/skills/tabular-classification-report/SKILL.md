---
name: tabular-classification-report
description: End-to-end classification on ANY tabular dataset (CSV/TSV/Parquet/Excel, a train/test pair, a folder or a .zip) that ends in a two-page LaTeX/PDF report. Profiles the data, lets the model choose and justify preprocessing and two contrasting classifiers, tunes and evaluates them with leakage-safe CV, reviews its own plan and report through checkpoints, and renders a report whose every number is traceable to script output. Use when asked to "analyse/classify this dataset", "compare two classifiers", "build a classification report", or for the IN6227 assignment.
argument-hint: <data path | train.csv | dataset.zip> [--test test.csv] [--target col] [--name run-name]
---

# Tabular classification -> two-page report

You are running a complete data-mining workflow. The **scripts compute facts; you make
decisions and write the argument**. Your value is the quality of each justification, not
the accuracy number: a well-argued simple model beats an unexplained complex one.

## Non-negotiable invariants

1. **The test set is touched once.** All choices (preprocessing, hyper-parameters, the
   decision threshold) are made with cross-validation on the training data. If a test
   result tempts you to change something, record that temptation in the oversight log
   instead of acting on it.
2. **Every preprocessing statistic is learned inside the CV fold.** This is guaranteed
   because `run_pipeline.py` wraps everything in an sklearn `Pipeline`; never preprocess
   the CSV yourself before the run.
3. **Numbers in the report come from JSON, never from your memory.** Prose refers to
   values with tokens (`[[R/models/lr/test/average_precision|3]]`). Tables are generated
   from `results.json`. Literal numbers in prose are listed in `numbers_audit.md` and must
   be checked at checkpoint 2.
4. **Every decision has a `why` that cites evidence** (a profile flag, a statistic, an
   ablation). "Standard practice" is not evidence.
5. **Claims match the statistics.** Say "significantly better" only if the paired test
   gives p < 0.05 *and* the bootstrap CI of the difference excludes 0; always discuss
   whether the difference is practically meaningful.
6. **Never hide a problem.** If the profiler raises `possible_target_leakage`,
   `train_test_overlap` or `id_like`, stop at checkpoint 1 and ask the human.

## Paths

`SKILL_DIR` = the directory containing this file. Scripts: `SKILL_DIR/scripts/`.
Run directory: `runs/<name>/` in the current project (default name: dataset file stem).
Python needs pandas, scikit-learn >= 1.3, scipy, matplotlib; PDF needs `tectonic`
(see `SKILL_DIR/references/setup.md`). Run scripts **from `SKILL_DIR/scripts`** so their
imports resolve, passing absolute paths for data and run directory.

## Workflow

Keep `runs/<name>/oversight_log.md` open for the whole run; append to it at every stage
(format in `references/oversight-protocol.md`). Start it with the invocation, the date,
the model name and the interface (e.g. `claude-opus-5-5`, Claude Code `claude --version`).

### Stage 1 - Profile (script)
```
python profile_data.py --data <DATA> [--test <TEST>] [--target <COL>] --out <RUN>
```
Read `<RUN>/profile.md` (digest) and, when needed, `profile.json`. Every entry in
`flags` needs a decision in Stage 2 (possibly "no action", with a reason).
If the target guess is wrong (profile says multiclass with hundreds of classes, or the
target looks like an ID), re-run with `--target`.

### Stage 2 - Plan (you)
Write `<RUN>/plan.json` following `references/plan-schema.md` and the rules in
`references/decision-rules.md`. In short:
- Address every flag; put the evidence in each `why`.
- Choose **exactly two** contrasting models: one interpretable/high-bias (usually
  LogisticRegression) and one flexible (usually HistGradientBoosting), unless the rules say
  otherwise. A prior-class baseline is added automatically.
- Decisions that only make sense for one model family go in that model's `preprocess`
  block (e.g. clip tails for LR, not for trees).
- Pick the primary CV metric from the class balance; define the threshold objective.
- Add 3-5 **ablations**, one per non-obvious decision, so each choice is tested rather than
  asserted.

### Checkpoint 1 - Plan review (human or reviewer agent)
Present the plan as a short table (decision | evidence | alternative rejected). Then:
- If the human is available, ask them to approve or edit it.
- If the human has delegated review, launch a **fresh reviewer sub-agent** with the
  prompt in `references/reviewer-prompts.md#plan`, giving it the paths of profile.md and
  plan.json only (not your reasoning) so the review is independent.
Log every reviewer point and your response (accepted -> what changed; rejected -> why).

### Stage 3 - Run (script)
```
python run_pipeline.py --data <DATA> [--test <TEST>] [--target <COL>] --plan <RUN>/plan.json --out <RUN>
```
Produces `results.json` and `fig_results.pdf`. Then inspect, in this order:
1. `params_at_search_boundary` non-empty -> widen that range **once** and re-run (log it).
2. `fit.stopped_early_or_converged == false` -> raise `max_iter` / check scaling.
3. `overfit_gap_primary` > 0.05 -> strengthen regularisation in the search space.
4. Ablations whose delta contradicts the plan's rationale -> keep the decision only if
   you can still defend it; either way, say so in the report (this is a finding, not a
   failure).
Re-runs triggered here use CV evidence only; never iterate on test metrics.

### Stage 4 - Write (you)
Write `<RUN>/narrative.json` following `references/narrative-guide.md`: five sections
(data, features, training, evaluation, findings), an issues table, a figure caption and
metadata. Use tokens for every number that exists in the JSON files; place the auto
tables with `%%results_table%%`, `%%config_table%%`, `%%ablation_table%%`,
`%%issues_table%%`, `%%figure%%`.

### Stage 5 - Render (script)
```
python render_report.py --run <RUN>
```
Exit 2 = unresolved token (fix the path; look it up in the JSON). Exit 3 = the main
report is longer than 2 pages: cut words, not content-bearing tables; never shrink fonts
below the template. Repeat until it passes.

### Checkpoint 2 - Report review (human or reviewer agent)
Launch a fresh reviewer with `references/reviewer-prompts.md#report`: it checks
`numbers_audit.md` literals against the JSON, spot-checks at least three table values,
and attacks the reasoning (unsupported claims, missing alternatives, over-reading small
differences). Fix, re-render, and log the outcome.

### Stage 6 - Reflection scaffold
If the deliverable needs a reflection (IN6227 Variant 2), draft `<RUN>/reflection.tex`
from the oversight log: human oversight (what was reviewed and changed), one decision
you find questionable (with evidence), and how numbers were verified (name a specific
value and the file it was checked against). Re-render; the reflection starts on a new
page and is excluded from the page check.

### Stage 7 - Hand-off
Report to the user: the PDF path, the two models and the headline comparison, the
decisions that ablations did **not** support, and anything in the oversight log that
still needs a human decision.

## Generalisation notes

- Works for binary and multiclass targets; for multiclass the primary metric is macro-F1,
  threshold tuning is skipped and per-class F1 replaces the PR curve.
- No test file -> a stratified 20 % hold-out is created (seeded) and reported as such.
- Label hygiene that is never a modelling choice (whitespace, `'>50K.'` vs `'>50K'`,
  missing labels) is handled in `common.py` and logged in `profile.json/load`.
- Very large data (> 200k rows): use 3 folds and smaller `n_iter` (see decision rules).
