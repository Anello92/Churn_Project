import joblib
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Customer Churn Prediction", page_icon="🔮")

@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('final_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler file not found. Please ensure 'final_model.pkl' and 'scaler.pkl' are in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model or scaler: {e}")
        st.stop()

model, scaler = load_model_and_scaler()

st.title('Customer Churn Prediction')
st.write("Enter customer details to predict churn probability.")

with st.form("prediction_form"):
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    monthly_usage = st.number_input('Monthly Usage (GB)', min_value=0, max_value=1000, value=50)
    customer_satisfaction = st.slider('Customer Satisfaction', min_value=0, max_value=10, value=5)
    monthly_value = st.number_input('Monthly Value ($)', min_value=0, max_value=500, value=50)
    plan = st.selectbox('Plan', ['Basic', 'Premium', 'Standard'])
    contract_duration = st.selectbox('Contract Duration', ['Short', 'Medium', 'Long'])

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    try:
        input_data = preprocess_input(age, monthly_usage, customer_satisfaction, monthly_value, plan, contract_duration)
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        st.subheader("Prediction Results")
        if prediction[0] == 1:
            st.warning("The customer is likely to churn.")
        else:
            st.success("The customer is likely to stay.")

        st.write(f"Probability of churning: {probability[0][1]:.2%}")

        st.subheader("Input Features")
        st.write(pd.DataFrame({
            'Feature': input_data.columns,
            'Value': input_data.iloc[0].values
        }))

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Please check your input values and try again.")

st.sidebar.title("About")
st.sidebar.info("This app predicts customer churn based on various factors. "
                "Enter the customer details and click 'Predict Churn' to see the results.")

st.sidebar.title("Model Information")
st.sidebar.info("This model uses a Random Forest classifier trained on historical customer data. "
                "The features used include age, monthly usage, customer satisfaction, monthly value, plan type, and contract duration.")