# Data profile digest

- task: **multiclass**, target `class`; train 1382 rows, test 346 rows (stratified 20% hold-out (seed 42)); 6 features (0 numeric, 6 categorical); one-hot width ~21
- target (train): {'unacc': 968, 'acc': 307, 'good': 55, 'vgood': 52}; minority share 0.0376
- rows with any missing feature: train 0, test 0
- duplicates: {'full_rows': 0, 'feature_dups_with_conflicting_label': 0, 'test_rows_also_in_train': 0}

## Numeric columns
| col | miss | min | q1 | med | q3 | max | skew | zeros | >3IQR | KS |
|---|---|---|---|---|---|---|---|---|---|---|

## Categorical columns
| col | miss | levels | dominant | rare(<1%) | placeholders | +rate by level |
|---|---|---|---|---|---|---|
| buying | 0/0 | 4 | 0.2598 | 0 (0.0) | [] |  |
| maint | 0/0 | 4 | 0.2576 | 0 (0.0) | [] |  |
| doors | 0/0 | 4 | 0.2598 | 0 (0.0) | [] |  |
| persons | 0/0 | 3 | 0.3466 | 0 (0.0) | [] |  |
| lug_boot | 0/0 | 3 | 0.3357 | 0 (0.0) | [] |  |
| safety | 0/0 | 3 | 0.3401 | 0 (0.0) | [] |  |

## Single-feature predictive power (CV AUC of depth-4 tree)
{'safety': 0.747, 'buying': 0.687, 'persons': 0.683, 'maint': 0.65, 'lug_boot': 0.572, 'doors': 0.465}

## Linear redundancy (R^2 from other numerics)
{}

## Flags
- **class_imbalance** `class`: minority share 0.0376, ratio 18.62:1
