# Financial Fraud Detection - Project Summary

Sentinel / Financial Fraud Intelligence

- Dataset: `transactions.csv` - 50,000 transactions, 8,963 users, 01 Jan 2023 to 31 Dec 2023
- Shipped fraud rate: 32.13% (16,067 of 50,000)
- Split: stratified 75/25, SMOTE applied to the training fold only.

## Headline result

On the shipped `Fraud_Label`, every model lands at ROC-AUC ~0.50 (range 0.492 to 0.506). That is the correct outcome for a label generated independently of the features, and it is reported as a data-quality finding rather than a modelling failure.

In the documented risk-model demo, the best estimator is **Logistic Regression** with ROC-AUC 0.874, PR-AUC 0.439 and F1 0.403 at the 0.5 threshold.

## Model comparison - risk-model demo

```
| model | roc_auc | pr_auc | f1 | precision | recall | accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.874 | 0.439 | 0.403 | 0.294 | 0.641 | 0.883 |
| Soft-Voting Ensemble | 0.869 | 0.409 | 0.421 | 0.455 | 0.392 | 0.934 |
| XGBoost | 0.864 | 0.423 | 0.321 | 0.646 | 0.214 | 0.944 |
| Random Forest | 0.858 | 0.413 | 0.325 | 0.614 | 0.221 | 0.944 |
| Support Vector Classifier | 0.830 | 0.302 | 0.370 | 0.314 | 0.452 | 0.906 |
| K-Nearest Neighbours | 0.826 | 0.305 | 0.313 | 0.204 | 0.669 | 0.819 |
| Decision Tree | 0.826 | 0.275 | 0.367 | 0.369 | 0.365 | 0.923 |
| Isolation Forest | 0.734 | 0.232 | 0.262 | 0.311 | 0.227 | 0.922 |
| AutoEncoder | 0.567 | 0.101 | 0.155 | 0.156 | 0.154 | 0.897 |
```

## Model comparison - shipped labels

```
| model | roc_auc | pr_auc | f1 | precision | recall | accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| Support Vector Classifier | 0.506 | 0.326 | 0.000 | 0.000 | 0.000 | 0.679 |
| Isolation Forest | 0.504 | 0.324 | 0.081 | 0.311 | 0.047 | 0.660 |
| Decision Tree | 0.502 | 0.321 | 0.023 | 0.288 | 0.012 | 0.673 |
| Soft-Voting Ensemble | 0.500 | 0.319 | 0.000 | 0.000 | 0.000 | 0.679 |
| K-Nearest Neighbours | 0.499 | 0.322 | 0.080 | 0.315 | 0.046 | 0.661 |
| AutoEncoder | 0.498 | 0.323 | 0.102 | 0.317 | 0.060 | 0.656 |
| Random Forest | 0.497 | 0.320 | 0.000 | 0.000 | 0.000 | 0.679 |
| XGBoost | 0.496 | 0.318 | 0.011 | 0.324 | 0.006 | 0.677 |
| Logistic Regression | 0.492 | 0.320 | 0.000 | 0.000 | 0.000 | 0.679 |
```

## Method

1. Exploratory analysis: class balance, amount distributions, fraud rate by channel.
2. Leakage-free preprocessing: ColumnTransformer (median impute + scale for numeric,
   most-frequent impute + one-hot for categorical) fitted on the training fold only.
3. Class imbalance: SMOTE on the training fold, never on the hold-out.
4. Nine estimators: logistic regression (L1), KNN, SVC (RBF), decision tree (depth-limited),
   random forest, XGBoost, isolation forest, MLP autoencoder, and a soft-voting ensemble.
5. Threshold analysis and confusion matrices on the untouched hold-out.

## Reproduce

```bash
pip install -r requirements.txt
streamlit run app.py
python scripts/generate_reports.py
```

## Limitations

- The shipped label is independent of the features; no model can rank it above chance.
- The demo target is synthetic and exists to exercise the scoring and alerting paths.
- Email/SMS delivery is mocked; no credentials are stored in the repository.
