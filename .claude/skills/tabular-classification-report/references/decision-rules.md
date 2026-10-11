# Decision rules

Each rule maps a profiler flag (or a fact in `profile.json`) to a default action and the
reasoning you should be ready to defend. Defaults are starting points: deviate when the
evidence says so, and write down why. Every non-obvious decision gets an ablation.

## 1. Cleaning

| Flag / fact | Default action | Why / when to deviate |
|---|---|---|
| `missing_label` | Rows dropped (done by loader) | A label cannot be imputed without inventing ground truth. If > 5 % of rows, discuss possible selection bias. |
| `missing_values` <= 5 % of rows | Numeric: median + missing indicator. Categorical: own level `__missing__` | Median is robust to the tails; the indicator lets the model use missingness if it is informative (MNAR). Own level avoids pretending the value is the mode. |
| `missing_values` 5-40 % | Same, and check the indicator's permutation importance | Missingness at this rate is often informative; say so if the indicator matters. |
| `missing_values` > 40 % | Consider `drop_columns`; keep if single-feature AUC of the column is high | A mostly-empty column is mainly its indicator. |
| `co_occurring_zeros` | `zero_as_missing` for the block + indicators; ablate | Several continuous measures being exactly 0 on the same rows looks like "not recorded". The indicator keeps the signal (check `target_dist_in_block`), the NaN removes an artificial point mass that bends linear fits. If zero is physically meaningful (counts, balances), keep it. |
| `isolated_zeros` (single column) | Usually keep; convert only with a domain argument | One column with zeros far from the bulk can be a sentinel, but also a real minimum. |
| `heavy_tail_or_outliers` | Never delete rows. Linear / distance models: `clip_quantiles` [0.005, 0.995] (or `log_transform` if skew >= 2 and values >= 0). Trees: nothing | The test set contains the same kind of extremes, so deleting them makes CV optimistic. Trees split on ranks and are invariant to monotone transforms. |
| `duplicate_rows` | Keep unless > 1 % and clearly artefactual | Duplicates across CV folds inflate scores slightly; removing natural repeats biases the distribution. Never touch the test set. |
| `train_test_overlap` | Stop -> ask the human | Possible leakage between the given splits. |
| `possible_target_leakage` | Stop -> ask the human; usually drop | A single feature with CV AUC >= 0.97 is almost always post-outcome information. |
| `id_like` | `drop_columns` | Unique per row: memorisable, no generalisable signal. |
| `train_test_drift` | Keep; mention in findings | CV cannot see drift, so test may be worse than CV. |

## 2. Features

| Flag / fact | Default action | Why |
|---|---|---|
| `exact_linear_dependency` | Drop the derived column for linear / distance models (model-level `preprocess.drop_columns`); keep it for trees; ablate both | For LR it is perfectly collinear: no new information, non-identifiable coefficients. Axis-aligned trees cannot build a linear combination themselves, so the pre-computed one can help them. |
| `high_correlation` (0.8-0.999) | Keep; rely on regularisation | Dropping loses information; but warn that LR coefficients and permutation importances are shared between correlated features. |
| `rare_levels` | `rare_level_threshold` 0.01 | Levels with < 1 % of rows give noisy one-hot coefficients; pooling also handles unseen test levels. |
| `high_cardinality` (> 50 levels) | Threshold 0.005-0.02 so that <= ~30 levels remain | Keeps one-hot width manageable; HGB native supports <= 255 levels. |
| `placeholder_levels` (Other/Unknown) | Keep as a level if its positive rate differs from the overall rate; mention it | Explicit "Unknown" is often informative (the reason it is unknown correlates with the outcome). |
| `numeric_low_cardinality` | Keep numeric unless codes are unordered | Ordinal codes keep order information; unordered codes should be treated as categorical (cast column to string — note it). |
| `near_constant` (>= 90 % one level) | Keep (pooled) | Cheap, and the rare remainder may still carry signal; check permutation importance. |

## 3. Model choice: exactly two contrasting models + prior baseline

The comparison should answer a question: *is the signal mostly additive, or are there
interactions / non-linearities worth the loss of transparency?* Default pair:

| Role | Default | Encoding / scaling | Stopping criterion |
|---|---|---|---|
| Interpretable, high-bias | `LogisticRegression` (lbfgs) | one-hot + StandardScaler | gradient tolerance 1e-4, `max_iter` 5000 (check `stopped_early_or_converged`) |
| Flexible, low-bias | `HistGradientBoostingClassifier` | `native` categorical (<= 255 levels), no scaling | `early_stopping=True`, 10 % internal validation, patience 30 on log-loss, `max_iter` 2000 |

Switch when:
- n_train < 1 000: replace HGB by `RandomForestClassifier` (`encoding: ordinal`,
  `n_estimators` 300 fixed — more trees do not over-fit, so there is no stopping rule; tune
  `max_depth`, `min_samples_leaf`, `max_features`). Early stopping needs data to hold out.
- Any categorical column has > 255 levels: HGB with `encoding: ordinal` (or pool harder).
- All features numeric, few (< 15) and n < 20 000: `KNeighborsClassifier` is a valid
  flexible alternative (one-hot + scale; tune `n_neighbors`, `weights`).
- Very wide one-hot (> 500 columns) and n > 100 000: LR with `solver: saga`.
Justify the pair in each model's `why`, and in the report say which other candidates you
rejected and why (e.g. SVM: no calibrated probabilities and O(n^2) training at this n).

### Search spaces (randomised search, scoring = primary metric)
- LR: `C` loguniform [1e-3, 1e2]; `class_weight` [null, "balanced"]; n_iter 20.
- HGB: `learning_rate` loguniform [0.02, 0.3]; `max_leaf_nodes` randint [8, 64];
  `min_samples_leaf` randint [10, 200] (scale up with n: up to n/50);
  `l2_regularization` loguniform [1e-4, 10]; `class_weight` [null, "balanced"]; n_iter 30.
- RF: `max_depth` [null, 8, 16, 32]; `min_samples_leaf` randint [1, 20];
  `max_features` ["sqrt", 0.5, 1.0]; n_iter 20.
- Budget: reduce `n_iter` so that n_train x n_iter x folds stays below ~2e7 row-fits for
  slow models. A best value at a range boundary (`params_at_search_boundary`) means the
  range was too narrow: widen it once.

## 4. Evaluation

| Fact | Choice | Why |
|---|---|---|
| binary, minority share < 0.35 | primary `average_precision`; threshold objective `f1` | AP is threshold-free and focuses on the minority class; accuracy rewards predicting the majority. |
| binary, balanced | primary `roc_auc`; threshold objective `balanced_accuracy` | Classes matter equally. |
| multiclass | primary `f1_macro`; no threshold | Each class weighs the same regardless of size. |
| n_train < 2 000 | 10 folds | Lower variance of the CV estimate. |
| n_train > 200 000 | 3 folds | Cost; each fold is already large. |
| otherwise | 5 stratified folds | Standard bias/variance trade-off; keeps class ratio per fold. |

Always reported automatically: prior baseline, CV mean ± sd, test metrics at 0.5 and at the
OOF-tuned threshold, 95 % bootstrap CIs, paired bootstrap difference, McNemar test on test
predictions, Nadeau-Bengio corrected t-test on CV folds, permutation importance (test set,
inspection only), confusion matrices.

Resampling (SMOTE etc.) is not used by default: `class_weight` plus threshold tuning
addresses imbalance without synthesising points, and is searched as a hyper-parameter so
the data decide whether it helps.
