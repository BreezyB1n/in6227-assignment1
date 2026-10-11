# IN6227 Assignment 1 — Variant 2: a reusable classification-report SKILL

A Claude Code skill that takes **any tabular classification dataset** and produces a two-page
LaTeX/PDF report covering data exploration and cleaning, feature engineering, model training,
evaluation and comparison, and findings. The scripts compute the facts and the LLM makes and
argues the decisions. Every number in the report can be traced to script output.

- **Skill:** [`.claude/skills/tabular-classification-report/SKILL.md`](.claude/skills/tabular-classification-report/SKILL.md)
- **Submitted report (with Reflection):** [`submission/IN6227-Assignment-1-Variant-2.pdf`](submission/IN6227-Assignment-1-Variant-2.pdf)
- **Audit trail of the assignment run:** [`runs/in6227/oversight_log.md`](runs/in6227/oversight_log.md)

## Design

```
profile_data.py ──► profile.json / profile.md      facts + flags (no decisions)
        │
   LLM writes plan.json                            every decision has a "why" citing evidence
        │
   checkpoint 1: human or fresh reviewer agent     logged in oversight_log.md
        │
run_pipeline.py ──► results.json, fig_results.pdf  leakage-safe sklearn Pipelines, CV-only selection,
        │                                          OOF threshold, bootstrap CIs, McNemar, corrected t-test,
        │                                          ablations, permutation importance
   LLM writes narrative.json                       numbers only as tokens [[R/models/lr/test/...]]
        │
render_report.py ──► report.pdf, numbers_audit.md  tables built from JSON, unresolved token = error,
        │                                          main report hard-limited to 2 pages
   checkpoint 2: numbers + claims review           logged
```

Each design choice targets a specific risk:

| Risk with LLM-driven analysis | Mechanism |
|---|---|
| Hallucinated or stale numbers | Prose numbers are tokens resolved from JSON; tables are generated; literals are listed for review |
| Test-set leakage / peeking | All preprocessing inside `Pipeline`; hyper-parameters and threshold chosen on CV / out-of-fold only |
| Asserted rather than tested decisions | Each non-obvious decision gets an ablation with paired fold differences |
| Over-claiming | Significance requires p < 0.05 **and** a CI excluding 0; reviewer checks claims |
| Generic, unexplained choices | `references/decision-rules.md` maps profiler flags to defaults *with reasons*; deviations must be justified |
| Silent failures | Convergence, early stopping, search-boundary and over-fitting gap diagnostics in `results.json` |

## Generalisation

The final code was run on three different datasets:

| Run | Data | What changed automatically |
|---|---|---|
| [`runs/in6227`](runs/in6227/report.pdf) | Assignment dataset: 31k rows, 7 numeric + 8 categorical, binary 24 % positive, train/test given | AP primary, OOF F1 threshold, LR vs HGB, structural-zero and exact-collinearity handling |
| [`examples/breast_cancer`](examples/breast_cancer/run/report.pdf) | 569 rows, 30 numeric, binary 37 %, no test file | Stratified hold-out, ROC-AUC primary, 10 folds, Random Forest (n < 1000), zeros kept (physically meaningful), log-transform for LR |
| [`examples/car`](examples/car/run/report.pdf) | 1,728 rows, 6 categorical, 4 classes (smallest 3.8 %) | Macro-F1 primary, no threshold, per-class F1 figure, HGB beats LR decisively (rule-based target) |

## Usage

```bash
pip install -r requirements.txt && brew install tectonic
# in Claude Code, from this repository:
/tabular-classification-report path/to/data.csv [--test test.csv] [--target label]
```

The scripts can also be run by hand (see SKILL.md, stages 1, 3 and 5). The course dataset is not
included; place `dataset.zip` anywhere and pass its path.

Model: Claude Opus 5.5 (`claude-opus-5-5`). Interface: Claude Code 2.1.278 (desktop app, agent harness).
