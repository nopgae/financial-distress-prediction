# Financial Distress Prediction with XGBoost

Binary classification model that predicts corporate financial distress using company financial statement ratios. Applies SMOTE to handle severe class imbalance, tunes XGBoost hyperparameters, and validates with Stratified K-Fold cross-validation.

## Tech Stack

- **Language:** Python 3
- **ML:** XGBoost, scikit-learn, imbalanced-learn (SMOTE)
- **Data:** pandas, numpy
- **Visualization:** seaborn, matplotlib

## Approach

1. **EDA** – Distribution analysis and correlation heatmap of 30+ financial features
2. **Preprocessing** – Median imputation for missing values, StandardScaler normalization, drop sparse columns (>50% missing)
3. **Class Balancing** – SMOTE oversampling to handle imbalanced distress vs. non-distress ratio
4. **Model** – XGBoost with tuned hyperparameters (`max_depth=10`, `learning_rate=0.01`, `subsample=0.8`)
5. **Validation** – 5-fold Stratified K-Fold cross-validation

## Features

Financial ratios extracted from income statements and balance sheets:

- Revenue, Gross Profit, R&D Expenses, Operating Income
- Net Income, Earnings before Tax, Income Tax
- Liquidity and solvency ratios

## Project Structure

```
.
├── ds310proj2.ipynb                     # Full pipeline notebook
├── train_data.csv                       # Training set (financial ratios + labels)
├── test_data.csv                        # Test set
├── submission.csv                       # Predicted labels
├── class_distribution.png              # Target class distribution
├── correlation_heatmap.png             # Feature correlation matrix
└── README.md
```

## How to Run

```bash
pip install pandas numpy matplotlib seaborn xgboost scikit-learn imbalanced-learn notebook
jupyter notebook ds310proj2.ipynb
```

## Results

- XGBoost with tuned parameters + SMOTE applied on balanced training data
- Evaluated using accuracy, precision, recall, F1-score, and Stratified K-Fold CV

## Relevance to Data Science / SCM Roles

Mirrors demand forecasting and anomaly detection problems in supply chain:

- Handling class-imbalanced data (rare events: disruptions, demand spikes)
- Feature engineering from structured tabular data
- Gradient boosting for classification — core technique in production ML pipelines

---

*Course: DS 310 – Data-Driven Modeling, Penn State University (Fall 2024)*
