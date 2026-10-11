# narrative.json - how to write the report

The report is graded on the **completeness of the workflow and the depth of reasoning**.
Two pages is tight: every sentence should state a decision, its evidence, or a finding.

## Structure

```jsonc
{
  "meta": {
    "title": "Data Mining Workflow Report: <dataset>",
    "course_line": "IN6227-Assignment-1", "variant": "Variant-2",
    "name": "<FULL NAME>", "matric": "<MATRIC NO>",
    "llm_model": "claude-opus-5-5",
    "llm_interface": "Agent harness: Claude Code 2.1.278",
    "repo_url": "https://github.com/<user>/<repo>"
  },
  "abstract": "2-3 sentences: task, two models, headline result with tokens.",
  "data": "LaTeX text ... %%issues_table%%",
  "features": "...",
  "training": "... %%config_table%%",
  "evaluation": "... %%results_table%% %%figure%% %%ablation_table%%",
  "findings": "...",
  "issues_table": [["Issue", "Evidence (tokens)", "Treatment + why"], ...],
  "figure_caption": "..."
}
```
Values are LaTeX (escape `_` as `\_`, `%` as `\%`). Feature names go in `\texttt{}`.

## Tokens

`[[S/path/to/value|fmt]]` with S = `R` (results.json), `P` (profile.json) or `L` (plan.json);
path segments separated by `/` (list indices allowed). Formats: `3` (decimals, default),
`pct1` (percent), `int` (thousands), `+3` (signed), `p` (p-value), `g`, `raw`.

Examples:
- `[[R/models/hgb/test/average_precision|3]]`
- `[[R/comparison/bootstrap_diff_B_minus_A/average_precision/ci/0|+3]]`
- `[[P/minority_share|pct1]]`, `[[P/n_train|int]]`
- `[[R/ablations/0/delta|+4]]`, `[[R/comparison/mcnemar/p_exact|p]]`

Only write a literal number when it is not in any JSON (e.g. a coefficient you derived).
Every literal is listed in `numbers_audit.md` and must be verified at checkpoint 2.

## What each section must contain (rubric mapping)

1. **Data exploration and cleaning** - size, task, class balance, missingness, outliers,
   suspicious structure (zeros, duplicates, derived columns), *and what you did about each and why*.
   The issues table carries most of this; prose adds the reasoning that does not fit.
2. **Feature selection / engineering** - what was dropped / pooled / transformed, per model
   if it differs, with the argument (e.g. collinearity hurts LR identifiability but not trees).
   Say what you deliberately did *not* do (e.g. no target encoding, no resampling) and why.
3. **Model training** - why these two models (the question the pair answers), rejected
   alternatives, CV design, search space and budget, primary metric, stopping criteria,
   threshold selection. The config table shows the selected values.
4. **Evaluation and comparison** - results table, figure, statistical tests, ablations.
   Interpret: statistical *and* practical significance; precision/recall trade-off at the
   chosen threshold; what the confusion matrices say about the errors.
5. **Findings and discussion** - what the comparison reveals about the data (additive vs
   interaction signal), which decisions mattered and which did not (ablations), limitations
   (single split, unknown costs, drift, anonymised features), what you would do next.

## Style
- Lead with the claim, then the evidence: "LR needs the composite dropped: it is an exact
  linear combination ([[...]])." Not: "Next, we considered correlations."
- Report negative results plainly: "Clipping did not help (delta [[...]], p = [[...]])".
- No hype words (robust, powerful, state-of-the-art). No claim without a number.
