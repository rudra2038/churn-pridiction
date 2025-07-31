# churn-pridiction
# Customer Churn Prediction using Logistic Regression

This project builds a machine learning model to predict whether a customer will churn or not, using logistic regression. The model is trained on customer data and evaluated for performance using common classification metrics. Final results are exported to a CSV file for further use, such as in Power BI dashboards.

---

## Dataset

- **Input file:** `customer_data.csv`
- The dataset should include:
  - **Target Column:** `Churn` (1 = Churned, 0 = Retained)
  - **Feature Columns:** Any mix of numerical or categorical variables representing customer characteristics and usage behavior.

---

## Requirements

Install the required Python libraries using:

```bash
pip install pandas numpy scikit-learn
