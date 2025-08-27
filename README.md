# Customer Churn Prediction App

This project is a **Customer Churn Prediction** application created using **Python**, **Streamlit**, and **XGBoost**. The app predicts whether a customer is likely to **churn** or **stay** based on their details.

---

## Features

- Predicts customer churn using an XGBoost model.  
- User-friendly interface with **Streamlit**.  
- Input all required customer details and get instant predictions.  
- Handles multiple features like demographics, account information, and usage data.  

---

## Evaluation Scores (XGBoost Model)

- **Accuracy:** 0.84  
- **Precision:** 0.78  
- **Recall:** 0.82  
- **F1 Score:** 0.80  

These scores indicate the model performs well in predicting customer churn with balanced precision and recall.

---

## Columns / Features

The model uses the following customer features:

- Customer ID  
- Gender  
- Senior Citizen  
- Partner  
- Dependents  
- Tenure  
- Phone Service  
- Multiple Lines  
- Internet Service  
- Online Security  
- Online Backup  
- Device Protection  
- Tech Support  
- Streaming TV  
- Streaming Movies  
- Contract  
- Paperless Billing  
- Payment Method  
- Monthly Charges  
- Total Charges  

> ⚠️ **Note:** All columns must be provided, and **Customer ID is mandatory** to avoid feature mismatch errors.

---

## How to Use

1. Download this repository.  
2. Open a terminal in the folder where `churn.py` is located.  
3. Run the app using:

```bash
streamlit run churn.py
