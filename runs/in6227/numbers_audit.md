# Numbers audit

## Token-resolved values (source of truth: JSON files)

| token | value | rendered |
|---|---|---|
| `[[P/minority_share|pct1]]` | 0.2399 | 24.0\% |
| `[[R/models/hgb/test/average_precision|3]]` | 0.7069929872199056 | 0.707 |
| `[[R/models/lr/test/average_precision|3]]` | 0.700121223324963 | 0.700 |
| `[[R/comparison/bootstrap_diff_B_minus_A/average_precision/ci/0|+3]]` | 0.0010266988551406926 | +0.001 |
| `[[R/comparison/bootstrap_diff_B_minus_A/average_precision/ci/1|+3]]` | 0.013962069227631237 | +0.014 |
| `[[P/load/train_rows_missing_label|int]]` | 3 | 3 |
| `[[P/load/test_rows_missing_label|int]]` | 1 | 1 |
| `[[P/minority_share|pct1]]` | 0.2399 | 24.0\% |
| `[[P/imbalance_ratio|2]]` | 3.17 | 3.17 |
| `[[P/missing/rows_with_any_missing_train|int]]` | 47 | 47 |
| `[[P/zero_blocks/0/rows|int]]` | 265 | 265 |
| `[[P/zero_blocks/1/rows|int]]` | 38 | 38 |
| `[[P/numeric/load_ratio/outlier_share_3iqr|pct1]]` | 0.0179 | 1.8\% |
| `[[P/numeric/activity_duration/outlier_share_3iqr|pct1]]` | 0.038 | 3.8\% |
| `[[L/models/0/preprocess/clip_quantiles/q/0|pct1]]` | 0.005 | 0.5\% |
| `[[L/models/0/preprocess/clip_quantiles/q/1|pct1]]` | 0.995 | 99.5\% |
| `[[P/linear_redundancy/composite_rank/formula|raw]]` | 0.006*aptitude_score + 0.004*performance_score | 0.006*aptitude\_score + 0.004*performance\_score |
| `[[P/linear_redundancy/composite_rank/formula_r2|7]]` | 0.9999999 | 0.9999999 |
| `[[P/categorical/region/rare_levels|int]]` | 17 | 17 |
| `[[P/categorical/region/rare_share_total|pct1]]` | 0.0482 | 4.8\% |
| `[[P/categorical/species/positive_rate_by_level/Unknown|pct1]]` | 0.087 | 8.7\% |
| `[[L/rare_level_threshold/value|pct0]]` | 0.01 | 1\% |
| `[[P/high_corr_pairs/0/r|2]]` | 0.891 | 0.89 |
| `[[P/load/raw_train_rows|int]]` | 31112 | 31{,}112 |
| `[[P/load/raw_test_rows|int]]` | 13334 | 13{,}334 |
| `[[P/n_flags|int]]` | 22 | 22 |
| `[[P/zero_blocks/0/rows|int]]` | 265 | 265 |
| `[[P/single_feature_auc/instrument|3]]` | 0.779 | 0.779 |
| `[[L/rare_level_threshold/value|pct0]]` | 0.01 | 1\% |
| `[[R/models/lr/n_features_after_preprocessing|int]]` | 73 | 73 |
| `[[R/models/hgb/n_features_after_preprocessing|int]]` | 22 | 22 |
| `[[R/models/lr/n_features_after_preprocessing|int]]` | 73 | 73 |
| `[[L/cv/folds|int]]` | 5 | 5 |
| `[[R/models/lr/n_candidates|int]]` | 20 | 20 |
| `[[R/models/hgb/n_candidates|int]]` | 30 | 30 |
| `[[R/models/lr/fit/n_iter|int]]` | 213 | 213 |
| `[[R/models/lr/fit/max_iter|int]]` | 5000 | 5{,}000 |
| `[[R/models/hgb/fit/n_iter|int]]` | 198 | 198 |
| `[[R/models/hgb/fit/max_iter|int]]` | 2000 | 2{,}000 |
| `[[R/models/lr/oof/threshold|2]]` | 0.32 | 0.32 |
| `[[R/models/hgb/oof/threshold|2]]` | 0.35 | 0.35 |
| `[[R/models/lr/overfit_gap_primary|3]]` | 0.003326864931372575 | 0.003 |
| `[[R/models/hgb/overfit_gap_primary|3]]` | 0.03742731364737639 | 0.037 |
| `[[R/baseline/test/average_precision|3]]` | 0.23760594014850372 | 0.238 |
| `[[R/comparison/bootstrap_diff_B_minus_A/average_precision/mean|+3]]` | 0.0070540078005356275 | +0.007 |
| `[[R/comparison/bootstrap_diff_B_minus_A/average_precision/p_two_sided|p]]` | 0.022 | 0.022 |
| `[[R/comparison/cv_corrected_ttest/mean_diff|+3]]` | 0.011334015683739151 | +0.011 |
| `[[R/comparison/cv_corrected_ttest/p|p]]` | 0.030772751133327846 | 0.031 |
| `[[R/comparison/mcnemar/A_right_B_wrong|int]]` | 220 | 220 |
| `[[R/comparison/mcnemar/A_wrong_B_right|int]]` | 337 | 337 |
| `[[R/comparison/mcnemar/A_wrong_B_right|int]]` | 337 | 337 |
| `[[R/comparison/mcnemar/p_exact|p]]` | 8.123376333851227e-07 | $<$0.001 |
| `[[R/comparison/bootstrap_diff_B_minus_A/f1/ci/0|+3]]` | -0.0014900361015838359 | -0.001 |
| `[[R/comparison/bootstrap_diff_B_minus_A/f1/ci/1|+3]]` | 0.010266876091753753 | +0.010 |
| `[[R/comparison/bootstrap_diff_B_minus_A/balanced_accuracy/ci/0|+3]]` | -0.006479037594054401 | -0.006 |
| `[[R/comparison/bootstrap_diff_B_minus_A/balanced_accuracy/ci/1|+3]]` | 0.00287231388583945 | +0.003 |
| `[[R/models/lr/oof/threshold|2]]` | 0.32 | 0.32 |
| `[[R/models/lr/test/at_0.5/f1|3]]` | 0.6243257351661736 | 0.624 |
| `[[R/models/lr/test/at_tuned/f1|3]]` | 0.6651245067356103 | 0.665 |
| `[[R/models/lr/test/at_0.5/precision|3]]` | 0.6956184567661885 | 0.696 |
| `[[R/models/lr/test/at_tuned/precision|3]]` | 0.5845491509208324 | 0.585 |
| `[[R/models/lr/test/at_0.5/recall|3]]` | 0.5662878787878788 | 0.566 |
| `[[R/models/lr/test/at_tuned/recall|3]]` | 0.7714646464646465 | 0.771 |
| `[[R/models/lr/test/at_tuned/confusion/1/0|int]]` | 724 | 724 |
| `[[R/models/hgb/test/at_tuned/confusion/1/0|int]]` | 793 | 793 |
| `[[P/target_test/yes|int]]` | 3168 | 3{,}168 |
| `[[R/ablations/5/delta|+4]]` | -0.0034948922722929376 | -0.0035 |
| `[[R/ablations/5/corrected_ttest/p|p]]` | 0.005040461543169888 | 0.005 |
| `[[R/ablations/2/delta|+4]]` | -0.0006132161298213479 | -0.0006 |
| `[[R/ablations/2/corrected_ttest/p|p]]` | 0.09311877126572454 | 0.093 |
| `[[R/models/hgb/permutation_importance/top/1/mean|3]]` | 0.05401529566615626 | 0.054 |
| `[[R/models/lr/permutation_importance/top/0/mean|3]]` | 0.2591017259507892 | 0.259 |
| `[[R/ablations/0/delta|+4]]` | 0.00022110919996753964 | +0.0002 |
| `[[R/ablations/1/delta|+4]]` | 0.0004212710165707456 | +0.0004 |
| `[[P/zero_blocks/0/rows|int]]` | 265 | 265 |
| `[[R/ablations/4/delta|+4]]` | 0.0002464148835168434 | +0.0002 |
| `[[P/n_train|int]]` | 31109 | 31{,}109 |
| `[[P/n_test|int]]` | 13333 | 13{,}333 |
| `[[P/minority_share|4]]` | 0.2399 | 0.2399 |
| `[[R/models/hgb/test/at_tuned/recall|4]]` | 0.7496843434343434 | 0.7497 |

## Literal numbers typed in prose (verify by hand)

- 0.004
- 0.006
- 0.009
- 0.5
- 10
- 100
- 114
- 15
- 2.1
- 21
- 2375
- 25.03
- 265
- 278
- 30
- 5.5
- 76
- 95
