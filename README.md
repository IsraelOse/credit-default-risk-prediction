# Credit Default Risk Prediction

Loan-default classification on an **85,000-application** consumer-credit dataset, delivered as a **modular, production-style pipeline** with a companion analysis notebook. Seed = 42.

## Problem
Predict whether a loan will default (`outcome_status == "Charged Off"`) from application-time attributes: amount, interest rate, term, income, debt-to-income, credit score, utilisation, employment tenure, risk grade, housing, income verification, loan purpose, and region.

## Repository architecture
Decoupled, block-based layout; each block is a pure step with type hints and `logging` (no `print`).

```
.
├── data_loaders/        # Read the applications + inquiries CSVs
│   └── credit_loader.py
├── transformers/        # Clean, split, preprocess, train/evaluate
│   ├── clean.py
│   ├── preprocessing.py
│   ├── split.py
│   └── model_training.py
├── data_exporters/      # Write metrics + feature importances
│   └── local_exporter.py
├── utils/               # Shared config + structured logging
├── pipelines/
│   └── credit_default/  # run.py + metadata.yaml + requirements.txt
├── Group6_Credit_Default_Risk_Notebook.ipynb   # Full EDA + modelling narrative
├── Group6_Credit_Default_Risk_Report.pdf        # Written report
├── datasets/            # applications (85k) + inquiries (45k)
└── outputs/             # Generated metrics + feature importance
```

## Run it
```bash
pip install -r pipelines/credit_default/requirements.txt
python -m pipelines.credit_default.run
```

## Approach
- **Cleaning:** parse text-encoded numerics (interest rate, term, utilisation, debt-to-income, employment tenure), derive the binary target, de-duplicate.
- **Leakage control:** only application-time features are modelled; outcome fields (status, last payment) are used solely to derive the target, never as predictors.
- **Preprocessing:** median-impute + scale numerics; mode-impute + one-hot encode categoricals, inside a `ColumnTransformer` fitted only on training data.
- **Models:** Logistic Regression and Random Forest, both class-balanced; compared on ROC-AUC.
- The 45k inquiries table has no join key to applications, so it is a secondary EDA source, not merged into the model.

## Results (held-out test set, verified)
| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|-------|:------:|:-------:|:--------:|:-----:|:--:|
| Logistic Regression | 0.723 | 0.662 | 0.659 | 0.673 | 0.666 |
| Random Forest | 0.724 | 0.662 | 0.659 | 0.673 | 0.666 |

The two models land within a thousandth of each other on ROC-AUC, indicating the signal is largely linear once cleaned. Interest rate is the strongest Random-Forest feature.

## Tech
Python · pandas · scikit-learn

*Project from the Master of Data Analytics program, University of Niagara Falls Canada.*
