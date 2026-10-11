# Data profile digest

- task: **binary**, target `label`; train 31109 rows, test 13333 rows (provided); 15 features (7 numeric, 8 categorical); one-hot width ~82
- target (train): {'no': 23645, 'yes': 7464}; minority share 0.2399
- rows with any missing feature: train 47, test 20
- duplicates: {'full_rows': 0, 'feature_dups_with_conflicting_label': 0, 'test_rows_also_in_train': 0}

## Numeric columns
| col | miss | min | q1 | med | q3 | max | skew | zeros | >3IQR | KS |
|---|---|---|---|---|---|---|---|---|---|---|
| aptitude_score | 4/1 | 90.0 | 310.59 | 333.51 | 402.56 | 510.0 | -0.292 | 0 | 0.0 | 0.007 |
| performance_score | 2/0 | 0.0 | 146.57 | 275.39 | 420.175 | 1010.0 | 0.552 | 265 | 0.0 | 0.008 |
| stability_index | 2/1 | 0.0 | 2.751 | 3.351 | 3.762 | 4.625 | -1.229 | 265 | 0.0 | 0.008 |
| load_ratio | 2/1 | 0.0 | 19.272 | 23.689 | 27.657 | 143.717 | 1.705 | 38 | 0.0179 | 0.009 |
| index_weight | 3/1 | 0.0 | 7.52 | 11.73 | 16.23 | 99.62 | 1.511 | 4 | 0.0052 | 0.007 |
| activity_duration | 2/1 | 0.0 | 72.74 | 80.33 | 90.26 | 205.0 | 0.199 | 38 | 0.038 | 0.008 |
| composite_rank | 1/4 | 0.589 | 2.591 | 3.199 | 3.838 | 6.876 | 0.258 | 0 | 0.0 | 0.006 |

## Categorical columns
| col | miss | levels | dominant | rare(<1%) | placeholders | +rate by level |
|---|---|---|---|---|---|---|
| geological_era | 2/2 | 5 | 0.8564 | 2 (0.0177) | [] | {'Triassic': 0.255, 'Jurassic': 0.12, 'Devonian': 0.251, 'Cambrian': 0.107, 'Permian': 0.123} |
| weather_pattern | 3/0 | 7 | 0.4589 | 1 (0.0008) | [] | {'Monsoon': 0.447, 'Heatwave': 0.044, 'Blizzard': 0.104, 'Fog': 0.064, 'Frost': 0.082, 'Drizzle': 0.106, 'Typhoon': 0.375} |
| region | 4/1 | 21 | 0.8974 | 17 (0.0482) | ['Other'] | {'Texas': 0.245, 'Arizona': 0.058, 'Wyoming': 0.248, 'Other': 0.202, 'South-Dakota': 0.219, 'Oregon': 0.29, 'Vermont': 0.383, 'Kentucky': 0.114} |
| personal_interest | 6/1 | 11 | 0.1274 | 0 (0.0) | ['Other', 'Unknown'] | {'Astronomy': 0.454, 'Beekeeping': 0.228, 'Falconry': 0.483, 'Archery': 0.138, 'Calligraphy': 0.266, 'Kite-flying': 0.043, 'Other': 0.218, 'Glassblowing': 0.119} |
| brew_preference | 7/3 | 2 | 0.668 | 0 (0.0) | [] | {'Espresso': 0.305, 'Latte': 0.11} |
| instrument | 3/2 | 6 | 0.4042 | 0 (0.0) | [] | {'Cello': 0.45, 'Oboe': 0.1, 'Dulcimer': 0.013, 'Theremin': 0.063, 'Zither': 0.475, 'Sitar': 0.035} |
| species | 3/2 | 7 | 0.6983 | 0 (0.0) | ['Unknown', 'Other'] | {'Lynx': 0.218, 'Heron': 0.282, 'Orca': 0.294, 'Unknown': 0.087, 'Viper': 0.28, 'Bison': 0.563, 'Other': 0.395} |
| mineral_type | 3/0 | 16 | 0.3239 | 2 (0.0065) | [] | {'Marble': 0.161, 'Opal': 0.193, 'Sapphire': 0.414, 'Ruby': 0.534, 'Granite': 0.249, 'Onyx': 0.05, 'Quartz': 0.246, 'Obsidian': 0.065} |

## Single-feature predictive power (CV AUC of depth-4 tree)
{'instrument': 0.779, 'weather_pattern': 0.77, 'composite_rank': 0.746, 'personal_interest': 0.722, 'aptitude_score': 0.709, 'mineral_type': 0.704, 'performance_score': 0.703, 'stability_index': 0.703, 'activity_duration': 0.664, 'brew_preference': 0.616, 'load_ratio': 0.585, 'species': 0.585, 'geological_era': 0.535, 'region': 0.517, 'index_weight': 0.517}

## Linear redundancy (R^2 from other numerics)
{'aptitude_score': {'r2': 1.0, 'formula': '-0.6667*performance_score + 166.7*composite_rank', 'formula_intercept': -0.00211, 'formula_r2': 0.9999995}, 'performance_score': {'r2': 1.0, 'formula': '-1.5*aptitude_score + 250*composite_rank', 'formula_intercept': -0.00338, 'formula_r2': 0.9999999}, 'stability_index': {'r2': 0.8296}, 'load_ratio': {'r2': 0.89022}, 'index_weight': {'r2': 0.00756}, 'activity_duration': {'r2': 0.85401}, 'composite_rank': {'r2': 1.0, 'formula': '0.006*aptitude_score + 0.004*performance_score', 'formula_intercept': 1e-05, 'formula_r2': 0.9999999}}

## Flags
- **missing_label** `label`: 3 train / 1 test rows dropped (cannot be imputed)
- **class_imbalance** `label`: minority share 0.2399, ratio 3.17:1
- **missing_values** `['aptitude_score', 'performance_score', 'stability_index', 'load_ratio', 'index_weight', 'activity_duration', 'composite_rank', 'geological_era', 'weather_pattern', 'region', 'personal_interest', 'brew_preference', 'instrument', 'species', 'mineral_type']`: train/test missing counts {'aptitude_score': '4/1', 'performance_score': '2/0', 'stability_index': '2/1', 'load_ratio': '2/1', 'index_weight': '3/1', 'activity_duration': '2/1', 'composite_rank': '1/4', 'geological_era': '2/2', 'weather_pattern': '3/0', 'region': '4/1', 'personal_interest': '6/1', 'brew_preference': '7/3', 'instrument': '3/2', 'species': '3/2', 'mineral_type': '3/0'}; worst column 0.02% of rows; rows with any missing: train 47
- **isolated_zeros** `stability_index`: 265 zeros (0.85%) far below the non-zero bulk
- **isolated_zeros** `load_ratio`: 38 zeros (0.12%) far below the non-zero bulk
- **heavy_tail_or_outliers** `load_ratio`: skew 1.705, 1.79% beyond 3*IQR, max 143.717
- **heavy_tail_or_outliers** `index_weight`: skew 1.511, 0.52% beyond 3*IQR, max 99.62
- **isolated_zeros** `activity_duration`: 38 zeros (0.12%) far below the non-zero bulk
- **heavy_tail_or_outliers** `activity_duration`: skew 0.199, 3.80% beyond 3*IQR, max 205.0
- **co_occurring_zeros** `['performance_score', 'stability_index']`: 265 rows zero in all; target distribution there {'no': 1.0}
- **co_occurring_zeros** `['load_ratio', 'activity_duration']`: 38 rows zero in all; target distribution there {'no': 0.842, 'yes': 0.158}
- **rare_levels** `geological_era`: 2 levels <1% each, 1.77% of rows in total
- **rare_levels** `weather_pattern`: 1 levels <1% each, 0.08% of rows in total
- **rare_levels** `region`: 17 levels <1% each, 4.82% of rows in total
- **placeholder_levels** `region`: explicit placeholder values ['Other']
- **placeholder_levels** `personal_interest`: explicit placeholder values ['Other', 'Unknown']
- **placeholder_levels** `species`: explicit placeholder values ['Unknown', 'Other']
- **rare_levels** `mineral_type`: 2 levels <1% each, 0.65% of rows in total
- **exact_linear_dependency** `aptitude_score`: aptitude_score = -0.6667*performance_score + 166.7*composite_rank + -0.00211 (R^2=0.9999995)
- **exact_linear_dependency** `performance_score`: performance_score = -1.5*aptitude_score + 250*composite_rank + -0.00338 (R^2=0.9999999)
- **exact_linear_dependency** `composite_rank`: composite_rank = 0.006*aptitude_score + 0.004*performance_score + 1e-05 (R^2=0.9999999)
- **high_correlation** `['composite_rank', 'performance_score', 'stability_index']`: 3 pairs with |r| >= 0.8; strongest: performance_score~stability_index r=0.891, performance_score~composite_rank r=0.88, stability_index~composite_rank r=0.83
