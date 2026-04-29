# Probabilistic vs Machine Learning Models for Predicting Customer Churn

Code for the CIS 4980 Senior BAS Capstone Thesis, University of Pennsylvania, Spring 2026.

**Anushka Daga** | Advisors: Peter S. Fader, Swapneel Sheth

## Overview

This repository contains all code used to produce the experiments and figures in the thesis. The core question is whether a hybrid framework that incorporates probabilistic survival model outputs as features in a machine learning classifier outperforms either approach alone for predicting customer churn.

Three model families are evaluated:
- **Probabilistic**: Weibull and Burr XII survival models
- **Machine learning**: Logistic Regression, Random Forest, XGBoost
- **Hybrid**: ML classifiers augmented with Burr XII survival features

## Data

The IBM Telco Customer Churn dataset is publicly available on [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn). Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` and place it in the project directory before running any scripts.

## Scripts

Run in this order:

| Script | Description |
|--------|-------------|
| `burr12_fit.py` | Fits Weibull and Burr XII survival models via MLE. Outputs `telco_burr12_features.csv` with survival features for each customer. |
| `eda.py` | Generates exploratory data analysis figures. |
| `ml_baseline.py` | Trains and evaluates baseline ML classifiers on static covariates (no tenure, no survival features). |
| `ml_hybrid.py` | Trains and evaluates hybrid models combining covariates with Burr XII survival features. |
| `sample_size.py` | Tests model performance across a logarithmic grid of training set sizes from n=20 to n=7032. |
| `tenure_ablation.py` | Compares ML with no tenure, ML with raw tenure, and hybrid without raw tenure. Tests whether hybrid gains reflect structured survival features or simply reintroduced tenure. |
| `final_results.py` | Generates the final holdout comparison bar charts across all model families. |

## Requirements

```bash
pip install pandas numpy scipy scikit-learn xgboost lifelines matplotlib
```

Python 3.9+

## Results Summary

| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| Weibull | 0.500 | 0.266 |
| Burr XII | 0.748 | 0.490 |
| Logistic Regression | 0.813 | 0.565 |
| Random Forest | 0.812 | 0.573 |
| XGBoost | 0.806 | 0.572 |
| Hybrid Logistic Regression | 0.840 | **0.658** |
| **Hybrid Random Forest** | **0.843** | 0.650 |
| Hybrid XGBoost | 0.834 | 0.658 |

All models trained on 80% of data and evaluated on the same fixed 20% held-out test set.

## Key Findings

- The Burr XII outperforms the Weibull entirely due to the heterogeneity parameter — a single additional parameter drives the entire performance gap from ROC-AUC 0.500 to 0.748
- Hybrid models consistently outperform both standalone approaches across all metrics
- Burr XII survival features match or beat raw tenure in all classifiers, confirming they encode structured nonlinear information rather than simply reintroducing tenure
- The Burr XII outperforms ML classifiers below ~100 training observations; the hybrid dominates across the full range
