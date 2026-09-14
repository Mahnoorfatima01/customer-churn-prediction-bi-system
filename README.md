# Customer Churn Prediction & Business Intelligence System

An end-to-end data science project that predicts telecom customer churn, explains *why* each customer is at risk, and presents everything through an interactive Streamlit dashboard.

## Business Problem

Telecom companies spend heavily to acquire customers, so losing existing ones directly hurts revenue. This project answers:
- Which customers are likely to leave?
- What factors drive churn?
- Which customer segments are highest-risk?
- What is the estimated revenue impact of churn?
- Which customers should be prioritized for retention?

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample dataset via Kaggle) — 7,043 customers, 21 features, including demographics, account info, subscribed services, billing, and churn status.

## Tech Stack

Python, Pandas, NumPy, Matplotlib, Seaborn, Plotly, Scikit-learn, XGBoost, SHAP, Streamlit

## Project Workflow

Raw Data → Cleaning → EDA → Feature Engineering → Train/Test Split → Model Training → Evaluation → Explainability → Risk Scoring → Dashboard

## Key Findings (from EDA)

- Overall churn rate: **26.54%**
- Month-to-month contracts churn at ~**43%**, vs. ~3% for two-year contracts
- New customers (first 1-2 months) churn at nearly **50%**
- Fiber optic internet customers churn at ~**42%**, the highest of any service type
- Electronic check payment users churn at ~**45%**, roughly 3x automatic payment methods
- Customers without Tech Support churn at ~**31%**, vs. ~15% with it

## Model Comparison

Five models were trained and evaluated on a held-out test set (20%, stratified):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7410 | 0.5076 | 0.8021 | 0.6218 | **0.8451** | 0.6513 |
| Gradient Boosting | 0.7999 | 0.6597 | 0.5080 | 0.5740 | 0.8431 | **0.6564** |
| Decision Tree | 0.7346 | 0.5000 | 0.8209 | 0.6215 | 0.8280 | 0.6040 |
| Random Forest | 0.7637 | 0.5476 | 0.6310 | 0.5863 | 0.8246 | 0.6112 |
| XGBoost | 0.7601 | 0.5385 | 0.6738 | 0.5986 | 0.8242 | 0.6192 |

**Selected model: Logistic Regression** — highest ROC-AUC and Recall, meaning it catches the most actual churners, which matters most given that a missed churner is more costly to the business than a false alarm. It's also fully interpretable, supporting the explainability requirement.

## Explainability

Global feature importance (Logistic Regression coefficients, cross-validated with SHAP) shows the strongest churn drivers are: lack of a long-term contract, fiber optic internet, short tenure, and electronic check payment. Per-customer SHAP explanations power the individual "why this customer" predictions in the dashboard.

## Dashboard

Built with Streamlit, includes 7 pages:
1. **Executive Overview** — total customers, churn rate, high-risk count, revenue at risk
2. **Customer Analysis** — demographics, tenure, contract, spending, services
3. **Churn Analysis** — interactive churn breakdowns by contract, tenure, payment, services
4. **Customer Prediction** — single-customer form + batch CSV upload for churn scoring
5. **Model Performance** — full metric comparison, confusion matrices, ROC curves
6. **Explainability** — global feature importance
7. **Retention Recommendations** — filterable high-risk customer list with suggested actions

## Project Structure