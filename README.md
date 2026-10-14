# Fraud Detection Analysis: Model & Data Evaluation

## Project Overview
This project aimed to build a machine learning classification model to detect fraudulent transactions within a highly imbalanced dataset. The dataset consisted of roughly 555,719 total rows, with an 80/20 train-test split resulting in approximately 110,718 normal transactions and only 426 actual fraud cases in the evaluation set.

## Methodology & Techniques Applied
Due to the severe class imbalance (approx. 260:1 ratio of Normal to Fraud), several advanced techniques were applied to force the models to recognize the minority fraud class:

1. **Random Forest with Class Weights:** Implemented `class_weight='balanced'` to heavily penalize misclassifications of the fraud class.
2. **XGBoost (Extreme Gradient Boosting):** Transitioned to a sequential boosting model utilizing the `scale_pos_weight` parameter set to the exact class ratio to aggressively target residual errors in fraud prediction.
3. **Custom Decision Thresholds:** Lowered the prediction probability threshold (e.g., to 0.15) to artificially boost recall.


## Results & Findings
Despite applying best-in-class techniques for imbalanced data, the models consistently failed to detect the fraud class:
- **Recall for Class 1 (Fraud):** 0.00 (or near zero)
- **Precision for Class 1 (Fraud):** 0.00
- **F1-Score for Class 1:** 0.00

Even when forced to train on perfectly balanced undersampled data, the model was unable to separate normal transactions from fraudulent ones.

## Conclusion
The mathematical failure of the models to learn from perfectly balanced data indicates that **the dataset itself fundamentally lacks predictive signal**. 

The current features (columns) do not contain sufficient variance or distinct patterns to differentiate a fraudulent transaction from a normal one. No amount of algorithm tuning or artificial resampling can extract a pattern that does not mathematically exist in the raw features.

## Recommendations for Next Steps
Machine learning modeling should be paused on this dataset. The primary recommendation is to invest in **Feature Engineering and Data Enrichment**:
1. **Behavioral Velocity:** Track how many transactions a specific user makes within short time windows (e.g., 10 minutes, 1 hour).
2. **Baseline Deviations:** Calculate how much a current transaction amount deviates from a specific user's historical average.
3. **Geographic/Device Mismatch:** Collect data on IP addresses, device IDs, or mismatches between billing and shipping zip codes.

Once these richer, high-signal data points are captured, modeling can resume.
