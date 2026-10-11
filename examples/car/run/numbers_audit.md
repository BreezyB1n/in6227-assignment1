# Numbers audit

## Token-resolved values (source of truth: JSON files)

| token | value | rendered |
|---|---|---|
| `[[P/minority_share|pct1]]` | 0.0376 | 3.8\% |
| `[[R/comparison/bootstrap_diff_B_minus_A/f1_macro/mean|+3]]` | 0.10801036927679662 | +0.108 |
| `[[R/comparison/bootstrap_diff_B_minus_A/f1_macro/ci/0|+3]]` | 0.051782838117072696 | +0.052 |
| `[[R/comparison/bootstrap_diff_B_minus_A/f1_macro/ci/1|+3]]` | 0.170414295930417 | +0.170 |
| `[[P/target_train/unacc|int]]` | 968 | 968 |
| `[[P/target_train/acc|int]]` | 307 | 307 |
| `[[P/target_train/good|int]]` | 55 | 55 |
| `[[P/target_train/vgood|int]]` | 52 | 52 |
| `[[P/imbalance_ratio|1]]` | 18.62 | 18.6 |
| `[[P/onehot_width_estimate|int]]` | 21 | 21 |
| `[[P/n_train|int]]` | 1382 | 1{,}382 |
| `[[P/n_test|int]]` | 346 | 346 |
| `[[P/n_flags|int]]` | 1 | 1 |
| `[[L/cv/folds|int]]` | 10 | 10 |
| `[[R/models/hgb/best_params/max_leaf_nodes|int]]` | 66 | 66 |
| `[[R/models/lr/fit/n_iter|int]]` | 606 | 606 |
| `[[R/models/hgb/fit/n_iter|int]]` | 242 | 242 |
| `[[R/models/lr/test/confusion/2/0|int]]` | 11 | 11 |
| `[[R/comparison/mcnemar/A_wrong_B_right|int]]` | 19 | 19 |
| `[[R/comparison/mcnemar/A_right_B_wrong|int]]` | 0 | 0 |
| `[[R/comparison/mcnemar/p_exact|p]]` | 3.814697265625e-06 | $<$0.001 |
| `[[R/models/lr/test/f1_macro|3]]` | 0.8938096090167987 | 0.894 |

## Literal numbers typed in prose (verify by hand)

- 128
- 2.1
- 2000
- 25
- 278
- 5.5
- 64
- 95
