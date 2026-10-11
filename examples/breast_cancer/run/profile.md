# Data profile digest

- task: **binary**, target `diagnosis`; train 455 rows, test 114 rows (stratified 20% hold-out (seed 42)); 30 features (30 numeric, 0 categorical); one-hot width ~30
- target (train): {'benign': 285, 'malignant': 170}; minority share 0.3736
- rows with any missing feature: train 0, test 0
- duplicates: {'full_rows': 0, 'feature_dups_with_conflicting_label': 0, 'test_rows_also_in_train': 0}

## Numeric columns
| col | miss | min | q1 | med | q3 | max | skew | zeros | >3IQR | KS |
|---|---|---|---|---|---|---|---|---|---|---|
| mean radius | 0/0 | 6.981 | 11.71 | 13.34 | 15.935 | 28.11 | 0.986 | 0 | 0.0 | 0.044 |
| mean texture | 0/0 | 9.71 | 16.335 | 18.9 | 21.83 | 39.28 | 0.647 | 0 | 0.0022 | 0.102 |
| mean perimeter | 0/0 | 43.79 | 75.235 | 86.18 | 104.5 | 188.5 | 1.037 | 0 | 0.0 | 0.053 |
| mean area | 0/0 | 143.5 | 420.4 | 546.4 | 795.5 | 2501.0 | 1.72 | 0 | 0.0088 | 0.049 |
| mean smoothness | 0/0 | 0.0625 | 0.0851 | 0.0952 | 0.1049 | 0.1634 | 0.576 | 0 | 0.0 | 0.106 |
| mean compactness | 0/0 | 0.0194 | 0.0637 | 0.0923 | 0.1303 | 0.3454 | 1.271 | 0 | 0.0022 | 0.076 |
| mean concavity | 0/0 | 0.0 | 0.0295 | 0.0594 | 0.1322 | 0.4268 | 1.46 | 11 | 0.0 | 0.067 |
| mean concave points | 0/0 | 0.0 | 0.0197 | 0.0333 | 0.0741 | 0.2012 | 1.204 | 11 | 0.0 | 0.053 |
| mean symmetry | 0/0 | 0.106 | 0.162 | 0.1799 | 0.1954 | 0.304 | 0.738 | 0 | 0.0022 | 0.078 |
| mean fractal dimension | 0/0 | 0.0502 | 0.0576 | 0.0613 | 0.0664 | 0.0974 | 1.259 | 0 | 0.0066 | 0.086 |
| radius error | 0/0 | 0.1115 | 0.2337 | 0.3274 | 0.485 | 2.873 | 3.181 | 0 | 0.0132 | 0.095 |
| texture error | 0/0 | 0.3602 | 0.8418 | 1.139 | 1.4755 | 4.885 | 1.809 | 0 | 0.0088 | 0.056 |
| perimeter error | 0/0 | 0.757 | 1.613 | 2.31 | 3.3835 | 21.98 | 3.517 | 0 | 0.0198 | 0.065 |
| area error | 0/0 | 6.802 | 17.855 | 24.87 | 45.39 | 542.2 | 5.497 | 0 | 0.044 | 0.071 |
| smoothness error | 0/0 | 0.0017 | 0.0051 | 0.0063 | 0.008 | 0.0311 | 2.459 | 0 | 0.0044 | 0.126 |
| compactness error | 0/0 | 0.0023 | 0.013 | 0.0202 | 0.032 | 0.1064 | 1.739 | 0 | 0.0088 | 0.054 |
| concavity error | 0/0 | 0.0 | 0.015 | 0.0259 | 0.0405 | 0.396 | 5.391 | 11 | 0.0154 | 0.072 |
| concave points error | 0/0 | 0.0 | 0.0076 | 0.0109 | 0.0145 | 0.0528 | 1.617 | 11 | 0.0066 | 0.111 |
| symmetry error | 0/0 | 0.0079 | 0.015 | 0.0187 | 0.023 | 0.079 | 2.245 | 0 | 0.0176 | 0.079 |
| fractal dimension error | 0/0 | 0.0009 | 0.0022 | 0.0031 | 0.0045 | 0.0298 | 4.085 | 0 | 0.0198 | 0.051 |
| worst radius | 0/0 | 7.93 | 13.02 | 14.92 | 19.185 | 36.04 | 1.116 | 0 | 0.0 | 0.064 |
| worst texture | 0/0 | 12.02 | 21.49 | 25.48 | 30.2 | 49.54 | 0.491 | 0 | 0.0 | 0.127 |
| worst perimeter | 0/0 | 50.41 | 84.135 | 97.65 | 127.0 | 251.2 | 1.141 | 0 | 0.0 | 0.075 |
| worst area | 0/0 | 185.2 | 516.15 | 683.4 | 1122.5 | 4254.0 | 1.903 | 0 | 0.0132 | 0.066 |
| worst smoothness | 0/0 | 0.0712 | 0.1145 | 0.1311 | 0.1466 | 0.2226 | 0.42 | 0 | 0.0 | 0.103 |
| worst compactness | 0/0 | 0.0273 | 0.1477 | 0.2167 | 0.3381 | 1.058 | 1.469 | 0 | 0.0044 | 0.074 |
| worst concavity | 0/0 | 0.0 | 0.1193 | 0.2298 | 0.3857 | 1.252 | 1.198 | 11 | 0.0022 | 0.067 |
| worst concave points | 0/0 | 0.0 | 0.0641 | 0.101 | 0.1621 | 0.291 | 0.495 | 11 | 0.0 | 0.056 |
| worst symmetry | 0/0 | 0.1565 | 0.2512 | 0.2823 | 0.3187 | 0.6638 | 1.503 | 0 | 0.0088 | 0.079 |
| worst fractal dimension | 0/0 | 0.055 | 0.0714 | 0.0801 | 0.0921 | 0.2075 | 1.739 | 0 | 0.0044 | 0.056 |

## Categorical columns
| col | miss | levels | dominant | rare(<1%) | placeholders | +rate by level |
|---|---|---|---|---|---|---|

## Single-feature predictive power (CV AUC of depth-4 tree)
{'worst perimeter': 0.953, 'worst radius': 0.952, 'worst area': 0.934, 'mean concave points': 0.929, 'worst concave points': 0.927, 'mean perimeter': 0.911, 'mean radius': 0.908, 'mean area': 0.908, 'mean concavity': 0.899, 'worst concavity': 0.873, 'area error': 0.864, 'perimeter error': 0.856, 'radius error': 0.843, 'mean compactness': 0.812, 'worst compactness': 0.805, 'worst texture': 0.768, 'concavity error': 0.732, 'concave points error': 0.718, 'mean texture': 0.711, 'worst smoothness': 0.696, 'worst symmetry': 0.686, 'mean smoothness': 0.676, 'mean symmetry': 0.649, 'compactness error': 0.644, 'worst fractal dimension': 0.636, 'fractal dimension error': 0.602, 'symmetry error': 0.568, 'mean fractal dimension': 0.552, 'smoothness error': 0.543, 'texture error': 0.538}

## Linear redundancy (R^2 from other numerics)
{'mean radius': {'r2': 0.99975, 'formula': '0.1378*mean perimeter + 0.0006848*mean area + -3.294*mean compactness + -1.359*mean concavity + 0.1894*radius error + -0.03899*perimeter error + 1.287*concavity error + 0.1514*worst radius + -0.01056*worst perimeter + -0.000482*worst area + 0.1452*worst compactness', 'formula_intercept': 0.52481, 'formula_r2': 0.9997183}, 'mean texture': {'r2': 0.91568}, 'mean perimeter': {'r2': 0.99976, 'formula': '6.511*mean radius + 0.001063*mean area + 26.68*mean compactness + 7.856*mean concavity + -2.055*radius error + 0.2089*perimeter error + 0.005556*area error + -6.59*concavity error + -0.6639*worst radius + 0.09185*worst perimeter + -2.02*worst compactness', 'formula_intercept': -2.50088, 'formula_r2': 0.9997197}, 'mean area': {'r2': 0.99713}, 'mean smoothness': {'r2': 0.89002}, 'mean compactness': {'r2': 0.98292}, 'mean concavity': {'r2': 0.98664}, 'mean concave points': {'r2': 0.98586}, 'mean symmetry': {'r2': 0.76937}, 'mean fractal dimension': {'r2': 0.93639}, 'radius error': {'r2': 0.98764}, 'texture error': {'r2': 0.75386}, 'perimeter error': {'r2': 0.98731}, 'area error': {'r2': 0.9765}, 'smoothness error': {'r2': 0.73965}, 'compactness error': {'r2': 0.93539}, 'concavity error': {'r2': 0.94391}, 'concave points error': {'r2': 0.91625}, 'symmetry error': {'r2': 0.81117}, 'fractal dimension error': {'r2': 0.90733}, 'worst radius': {'r2': 0.99875}, 'worst texture': {'r2': 0.94558}, 'worst perimeter': {'r2': 0.99762}, 'worst area': {'r2': 0.99693}, 'worst smoothness': {'r2': 0.91684}, 'worst compactness': {'r2': 0.97317}, 'worst concavity': {'r2': 0.9696}, 'worst concave points': {'r2': 0.97418}, 'worst symmetry': {'r2': 0.89653}, 'worst fractal dimension': {'r2': 0.95204}}

## Flags
- **heavy_tail_or_outliers** `mean area`: skew 1.72, 0.88% beyond 3*IQR, max 2501.0
- **heavy_tail_or_outliers** `mean fractal dimension`: skew 1.259, 0.66% beyond 3*IQR, max 0.0974
- **heavy_tail_or_outliers** `radius error`: skew 3.181, 1.32% beyond 3*IQR, max 2.873
- **heavy_tail_or_outliers** `texture error`: skew 1.809, 0.88% beyond 3*IQR, max 4.885
- **heavy_tail_or_outliers** `perimeter error`: skew 3.517, 1.98% beyond 3*IQR, max 21.98
- **heavy_tail_or_outliers** `area error`: skew 5.497, 4.40% beyond 3*IQR, max 542.2
- **heavy_tail_or_outliers** `smoothness error`: skew 2.459, 0.44% beyond 3*IQR, max 0.0311
- **heavy_tail_or_outliers** `compactness error`: skew 1.739, 0.88% beyond 3*IQR, max 0.1064
- **heavy_tail_or_outliers** `concavity error`: skew 5.391, 1.54% beyond 3*IQR, max 0.396
- **heavy_tail_or_outliers** `concave points error`: skew 1.617, 0.66% beyond 3*IQR, max 0.0528
- **heavy_tail_or_outliers** `symmetry error`: skew 2.245, 1.76% beyond 3*IQR, max 0.079
- **heavy_tail_or_outliers** `fractal dimension error`: skew 4.085, 1.98% beyond 3*IQR, max 0.0298
- **heavy_tail_or_outliers** `worst area`: skew 1.903, 1.32% beyond 3*IQR, max 4254.0
- **heavy_tail_or_outliers** `worst symmetry`: skew 1.503, 0.88% beyond 3*IQR, max 0.6638
- **co_occurring_zeros** `['mean concavity', 'mean concave points', 'concavity error', 'concave points error', 'worst concavity', 'worst concave points']`: 11 rows zero in all; target distribution there {'benign': 1.0}
- **exact_linear_dependency** `mean radius`: mean radius = 0.1378*mean perimeter + 0.0006848*mean area + -3.294*mean compactness + -1.359*mean concavity + 0.1894*radius error + -0.03899*perimeter error + 1.287*concavity error + 0.1514*worst radius + -0.01056*worst perimeter + -0.000482*worst area + 0.1452*worst compactness + 0.52481 (R^2=0.9997183)
- **exact_linear_dependency** `mean perimeter`: mean perimeter = 6.511*mean radius + 0.001063*mean area + 26.68*mean compactness + 7.856*mean concavity + -2.055*radius error + 0.2089*perimeter error + 0.005556*area error + -6.59*concavity error + -0.6639*worst radius + 0.09185*worst perimeter + -2.02*worst compactness + -2.50088 (R^2=0.9997197)
- **high_correlation** `['area error', 'compactness error', 'concavity error', 'fractal dimension error', 'mean area', 'mean compactness', 'mean concave points', 'mean concavity', 'mean perimeter', 'mean radius', 'mean smoothness', 'mean texture', 'perimeter error', 'radius error', 'worst area', 'worst compactness', 'worst concave points', 'worst concavity', 'worst fractal dimension', 'worst perimeter', 'worst radius', 'worst smoothness', 'worst texture']`: 43 pairs with |r| >= 0.8; strongest: mean radius~mean perimeter r=0.998, worst radius~worst perimeter r=0.994, mean radius~mean area r=0.987, mean perimeter~mean area r=0.986
