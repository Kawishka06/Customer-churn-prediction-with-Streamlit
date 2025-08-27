#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

st.title("Customer Churn Prediction using Machine Learning")
# In[ ]:


df=pd.read_csv("Telco-Customer-Churn.csv")




# In[ ]:


# In[ ]:
df.head()


# In[ ]:


df.tail()


# In[ ]:
df.describe()

# In[ ]:
df.info()

# In[ ]:
df.isnull().sum()

# Drop the customer Id since it is not useful for predictions

# In[ ]:
df['Churn'].value_counts()


# In[ ]:


if 'customerId' in df.columns:
    df.drop('customerId',axis=1,inplace=True)


# Convert total charges to numeric

# In[ ]:


df["TotalCharges"]=pd.to_numeric(df['TotalCharges'],errors='coerce')
df.fillna(0,inplace=True)


# Encoding the Categorical Columns

# In[ ]:


lable={}
for col in df.select_dtypes(include=['object']).columns:
    le=LabelEncoder()
    df[col]=le.fit_transform(df[col])
    lable[col]=le


# In[ ]:


X=df.drop('Churn',axis=1)
y=df['Churn']


# Splitting the training and testing data

# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)


# In[ ]:


smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X, y)


# In[ ]:


y_train_res.value_counts()


# Training the model using Random Forest Classifier




# In[ ]:


xgb_model = XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_res, y_train_res)


# In[ ]:


y_pred_xgb = xgb_model.predict(X_test)

st.write("Model Training Complete")

# In[ ]:


xgb_model_accuracy=accuracy_score(y_test,y_pred_xgb)
xgb_model_precision=precision_score(y_test,y_pred_xgb)
xgb_model_recall=recall_score(y_test,y_pred_xgb)
xgb_model_f1score=f1_score(y_test,y_pred_xgb)

st.write("Accuracy : ",xgb_model_accuracy)
st.write("Precision: ",xgb_model_precision)
st.write("Model Recall Score: ",xgb_model_recall)
st.write("Model F1 Score: ",xgb_model_f1score)


# In[ ]:
st.subheader("Enter Customer Details")


# In[ ]:

customerID = st.text_input("Customer ID")
gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependent = st.selectbox("Dependent", ["No", "Yes"])
tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, step=1)
phone_service = st.selectbox("Phone Service", ["No", "Yes"])
multiple = st.selectbox("Multiple Lines", ["No phone", "No", "Yes"])
internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
online_security = st.selectbox("Online Security", ["No internet", "No", "Yes"])
online_backup = st.selectbox("Online Backup", ["No internet", "No", "Yes"])
device_protection = st.selectbox("Device Protection", ["No internet", "No", "Yes"])
tech_sup = st.selectbox("Tech Support", ["No internet", "No", "Yes"])
streaming_tv = st.selectbox("Streaming TV", ["No internet", "No", "Yes"])
streaming_movies = st.selectbox("Streaming Movies", ["No internet", "No", "Yes"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two years"])
paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
payment = st.selectbox("Payment Method", ["Electronic", "Mailed check", "Bank transaction", "Credit card"])
monthly_charge = st.number_input("Monthly Charge", min_value=0.0, max_value=1000.0, step=1.0)
total_charge = st.number_input("Total Charge", min_value=0.0, max_value=10000.0, step=1.0)

# ----------- Map inputs to numeric values ----------- #
multiple_mapping = {'no phone':0, 'no':1, 'yes':2}
internet_mapping = {'no':0, 'dsl':2, 'fiber optic':1}
yes_no_mapping = {'no':0, 'yes':2, 'no internet':1}
contract_mapping = {'month-to-month':0, 'one year':1, 'two years':2}
payment_mapping = {'electronic':3, 'mailed check':2, 'bank transaction':1, 'credit card':0}

gender_low = 1 if gender.lower() == 'male' else 0
senior = 1 if senior.lower() == 'yes' else 0
partner_low = 1 if partner.lower() == 'yes' else 0
dependent_low = 1 if dependent.lower() == 'yes' else 0
phone_service_low = 1 if phone_service.lower() == 'yes' else 0
multiple_low = multiple_mapping[multiple.lower()]
internet_low = internet_mapping[internet_service.lower()]
online_security_low = yes_no_mapping[online_security.lower()]
online_backup_low = yes_no_mapping[online_backup.lower()]
device_protection_low = yes_no_mapping[device_protection.lower()]
tech_sup_low = yes_no_mapping[tech_sup.lower()]
streaming_tv_low = yes_no_mapping[streaming_tv.lower()]
streaming_movies_low = yes_no_mapping[streaming_movies.lower()]
contract_low = contract_mapping[contract.lower()]
paperless_low = 1 if paperless.lower() == 'yes' else 0
payment_low = payment_mapping[payment.lower()]

# ----------- Prediction ----------- #
user_input = np.array([[0,gender_low, senior, partner_low, dependent_low, tenure, phone_service_low, 
                        multiple_low, internet_low, online_security_low, online_backup_low, device_protection_low, 
                        tech_sup_low, streaming_tv_low, streaming_movies_low, contract_low, paperless_low, 
                        payment_low, monthly_charge, total_charge]], dtype=float)

if st.button("Predict Churn"):
    prediction = xgb_model.predict(user_input)
    probability = xgb_model.predict_proba(user_input)[0]

    st.subheader("Prediction Result")
    if prediction[0]==1:
        st.warning("The customer is likely to churn the service.")
    else:
        st.success("The customer is unlikely to churn the service.")
    
    st.subheader("Probability")
    labels = ['Unlikely to churn', 'Likely to churn']
    sizes = [probability[0], probability[1]]
    colors = ["#ffacee","#A6B7FF"]
    
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
           colors=colors, explode=[0.1, 0], wedgeprops={'edgecolor':'black', 'linewidth':1.5})
    ax.set_title('Probability of Churning a Telephone Service')
    st.pyplot(fig)

