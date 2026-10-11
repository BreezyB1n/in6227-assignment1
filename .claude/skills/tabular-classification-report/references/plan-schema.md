# plan.json schema

Every decision is an object with the chosen value **and** a `why`. Model-level
`preprocess` entries override the dataset-level ones for that model only.

```jsonc
{
  "seed": 42,
  "positive_class": "yes",                       // binary only; default = minority class
  "positive_class_why": "...",
  "missing_label":       {"value": "drop", "why": "..."},          // informational (loader drops)
  "missing_indicators":  {"value": true, "why": "..."},            // SimpleImputer(add_indicator)
  "categorical_missing": {"value": "own_level" | "most_frequent", "why": "..."},
  "zero_as_missing":     {"columns": ["col", ...], "why": "..."},  // exact 0 -> NaN before imputation
  "clip_quantiles":      {"columns": [...], "q": [0.005, 0.995], "why": "..."}, // fitted per fold
  "log_transform":       {"columns": [...], "why": "..."},         // log1p, negatives clipped to 0
  "rare_level_threshold":{"value": 0.01, "why": "..."},            // pool levels below this share
  "placeholder_levels":  {"value": "keep", "why": "..."},          // informational
  "drop_columns":        {"columns": [...], "why": "..."},
  "cv": {"folds": 5, "primary_metric": "average_precision", "why": "..."},
  // binary metrics: average_precision roc_auc f1 f1_macro balanced_accuracy accuracy precision recall
  // multiclass:     f1_macro balanced_accuracy accuracy roc_auc_ovr neg_log_loss
  "threshold": {"strategy": "oof" | "none", "objective": "f1" | "f1_macro" | "balanced_accuracy", "why": "..."},
  "models": [
    {
      "id": "lr", "label": "Logistic Regression",
      "estimator": "LogisticRegression",
      // one of LogisticRegression DecisionTreeClassifier RandomForestClassifier ExtraTreesClassifier
      //        HistGradientBoostingClassifier KNeighborsClassifier GaussianNB
      "encoding": "onehot" | "ordinal" | "native",   // native = HGB categorical support
      "scale": true,
      "preprocess": { "drop_columns": {...}, "clip_quantiles": {...} },  // optional overrides
      "params": {"max_iter": 5000},                  // fixed parameters
      "search": {                                    // randomised search space
        "C": {"loguniform": [0.001, 100]},           // also {"uniform": [a,b]}, {"randint": [a,b]}
        "class_weight": [null, "balanced"]           // a list = categorical choice
      },
      "n_iter": 20,
      "why": "role in the comparison + stopping criterion"
    }
  ],
  "ablations": [
    // change: overrides merged into this model's preprocess; model_change: overrides model keys
    {"id": "A1_keep_zeros", "model": "lr", "change": {"zero_as_missing": {"columns": []}},
     "question": "Does treating co-occurring zeros as missing help?"},
    {"id": "A5_hgb_onehot", "model": "hgb", "model_change": {"encoding": "onehot"},
     "question": "Native categorical splits vs one-hot?"}
  ]
}
```

The first two entries of `models` are the compared pair (statistical tests are run on
them). Ablations reuse the tuned hyper-parameters of their model and report the change in
CV primary metric with a corrected resampled t-test.
