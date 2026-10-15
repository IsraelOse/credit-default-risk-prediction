# Credit Default Risk Prediction

Predicting loan default from a two-table credit dataset: an **85,000-row applications** file joined with a **45,000-row credit-inquiries** file.

## Approach
- Merged applications with aggregated inquiry history
- Data cleaning and, importantly, **target-leakage removal** (dropping fields that would not be known at decision time)
- Trained and compared Random Forest and Logistic Regression
- Evaluated with ROC-AUC on a held-out set

## Results
| Model | ROC-AUC |
|-------|---------|
| Random Forest | **0.726** |
| Logistic Regression | 0.722 |

The two models land close together, suggesting the signal is largely linear once leakage is removed.

## Repo layout
| Path | What it is |
|------|-----------|
| `datasets/` | Applications and inquiries CSVs |
| `Group6_Credit_Default_Risk_Notebook.ipynb` | Full analysis |
| `Group6_Credit_Default_Risk_Report.pdf` | 14-page report with figures |

## Tech
Python · pandas · scikit-learn

*Project from the Master of Data Analytics program, University of Niagara Falls Canada.*
