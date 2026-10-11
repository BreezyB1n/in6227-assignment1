# Numbers audit

## Token-resolved values (source of truth: JSON files)

| token | value | rendered |
|---|---|---|
| `[[P/n_features|int]]` | 30 | 30 |
| `[[R/comparison/bootstrap_diff_B_minus_A/roc_auc/mean|+3]]` | 0.001057461680272121 | +0.001 |
| `[[R/comparison/bootstrap_diff_B_minus_A/roc_auc/ci/0|+3]]` | -0.003764069264069228 | -0.004 |
| `[[R/comparison/bootstrap_diff_B_minus_A/roc_auc/ci/1|+3]]` | 0.007722007722007598 | +0.008 |
| `[[P/load/test_source|raw]]` | stratified 20% hold-out (seed 42) | stratified 20\% hold-out (seed 42) |
| `[[P/numeric/area error/skew|2]]` | 5.497 | 5.50 |
| `[[P/zero_blocks/0/rows|int]]` | 11 | 11 |
| `[[P/linear_redundancy/mean perimeter/formula_r2|4]]` | 0.9997197 | 0.9997 |
| `[[P/high_corr_pairs/0/r|3]]` | 0.998 | 0.998 |
| `[[P/n_train|int]]` | 455 | 455 |
| `[[P/n_test|int]]` | 114 | 114 |
| `[[P/minority_share|pct1]]` | 0.3736 | 37.4\% |
| `[[P/n_flags|int]]` | 18 | 18 |
| `[[L/cv/folds|int]]` | 10 | 10 |
| `[[R/models/lr/fit/n_iter|int]]` | 35 | 35 |
| `[[R/comparison/mcnemar/A_right_B_wrong|int]]` | 2 | 2 |
| `[[R/comparison/mcnemar/A_wrong_B_right|int]]` | 2 | 2 |
| `[[R/comparison/mcnemar/p_exact|p]]` | 1.0 | 1.000 |
| `[[R/comparison/cv_corrected_ttest/p|p]]` | 0.2346003969663633 | 0.235 |
| `[[P/n_test|int]]` | 114 | 114 |

## Literal numbers typed in prose (verify by hand)

- 0.01
- 10
- 1000
- 114
- 2.1
- 2.4
- 20
- 2000
- 278
- 300
- 35
- 45
- 5.5
- 95
