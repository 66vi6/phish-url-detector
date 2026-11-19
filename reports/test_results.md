# Test-set results

Test set: 32,172 URLs (11,058 phishing, 21,114 benign; 34.4% phishing, from 17,048 domains). Models were trained on train only; thresholds were chosen on validation.

## Headline (95% CIs from resampling whole domains, 1000 resamples)

| Model | Val PR-AUC | Test PR-AUC (95% CI) | Test ROC-AUC | Recall @1% FPR (95% CI) | Recall @0.1% FPR |
|---|---|---|---|---|---|
| Rule baseline | 0.3750 | 0.3515 (0.3104-0.3885) | 0.5112 | 0.001 (0.000-0.001) | 0.001 |
| n-gram LR | 0.9636 | 0.9661 (0.9613-0.9705) | 0.9791 | 0.734 (0.710-0.761) | 0.435 |
| LightGBM | 0.8977 | 0.9270 (0.9183-0.9349) | 0.9587 | 0.516 (0.455-0.572) | 0.207 |
| Blend | 0.9661 | 0.9708 (0.9667-0.9743) | 0.9832 | 0.743 (0.717-0.768) | 0.488 |

Blend minus n-gram LR (paired, PR-AUC): +0.0045 (95% CI +0.0030 to +0.0063); the CI excludes 0, so the gain is real.

Recall at a fixed FPR above uses the test set's own ROC curve, which is slightly optimistic. The next table is the realistic version.

## Operating points: threshold chosen on validation, applied to test

| Model | Target FPR | Test FPR | Test recall | Precision @0.1% prev | @1% prev | @5% prev |
|---|---|---|---|---|---|---|
| n-gram LR | 1.0% | 0.69% | 0.700 | 0.093 | 0.507 | 0.843 |
| n-gram LR | 0.1% | 0.08% | 0.411 | 0.352 | 0.846 | 0.966 |
| LightGBM | 1.0% | 1.06% | 0.528 | 0.047 | 0.335 | 0.724 |
| LightGBM | 0.1% | 0.19% | 0.310 | 0.141 | 0.623 | 0.896 |
| Blend | 1.0% | 0.73% | 0.717 | 0.089 | 0.497 | 0.837 |
| Blend | 0.1% | 0.07% | 0.444 | 0.385 | 0.863 | 0.971 |

Precision columns recompute precision for a realistic share of phishing in traffic (0.1%, 1%, 5%). The test set itself is much more phishing-heavy than real traffic, which makes its raw precision misleading.
