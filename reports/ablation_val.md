# Ablation (LightGBM, validation set)

| Setting | # features | Val PR-AUC | Recall @1% FPR | Change vs all |
|---|---|---|---|---|
| All features | 30 | 0.8977 | 0.511 | +0.0000 |
| Drop size | 26 | 0.8944 | 0.504 | -0.0033 |
| Drop structure | 23 | 0.8660 | 0.395 | -0.0317 |
| Drop host_tricks | 26 | 0.8966 | 0.501 | -0.0011 |
| Drop randomness | 25 | 0.8753 | 0.475 | -0.0224 |
| Drop keywords | 27 | 0.8871 | 0.487 | -0.0107 |
| Drop brand | 25 | 0.8949 | 0.505 | -0.0028 |
| Drop misc | 28 | 0.8961 | 0.495 | -0.0016 |
| Only size | 4 | 0.7927 | 0.316 | -0.1050 |
| Only structure | 7 | 0.8290 | 0.415 | -0.0688 |
| Only host_tricks | 4 | 0.3674 | 0.010 | -0.5304 |
| Only randomness | 5 | 0.8098 | 0.261 | -0.0880 |
| Only keywords | 3 | 0.3861 | 0.001 | -0.5117 |
| Only brand | 5 | 0.3570 | 0.007 | -0.5407 |
| Only misc | 2 | 0.3774 | 0.031 | -0.5203 |
