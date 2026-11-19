# Validation results

| Model | PR-AUC | ROC-AUC | Recall @0.1% FPR | Recall @1% FPR | Precision @1% prevalence (0.1% FPR) | Notes |
|---|---|---|---|---|---|---|
| Rule baseline | 0.3750 | 0.5178 | 0.001 | 0.001 | 1.000 | flag if any red flag: precision 0.59, recall 0.06 |
| TF-IDF char n-grams + LR | 0.9636 | 0.9765 | 0.495 | 0.722 | 0.839 | C=16 |
| LightGBM (handcrafted features) | 0.8977 | 0.9235 | 0.341 | 0.511 | 0.782 | 207 trees |
