# External check: held-out PhishTank phishing

| Step | Rows |
|---|---|
| PhishTank rows | 43,811 |
| after exact dedupe | 43,396 |
| after template dedupe | 41,120 |
| after removing URLs seen in train/val | 36,298 |
| after removing domains seen in train/val | 6,883 |

PhreshPhish phishing dates: 2024-07-02 to 2025-11-06. PhishTank dates: 2013-05-28 to 2026-11-18. This test mixes a new feed with a later time period, and the two effects can't be separated.

Thresholds are the validation thresholds at 1% FPR. Negatives for AUC are the test-split benign URLs.

| Model | Recall on PhishTank (95% CI) | Recall on PhreshPhish test phish | Test FPR | ROC-AUC (PhishTank vs test benign) |
|---|---|---|---|---|
| n-gram LR | 0.685 (0.674-0.696) | 0.700 | 0.69% | 0.9801 |
| LightGBM | 0.587 (0.575-0.599) | 0.528 | 1.06% | 0.9583 |
| Blend | 0.702 (0.691-0.713) | 0.717 | 0.73% | 0.9827 |
