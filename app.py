import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("churn_model_bundle_deploy.joblib")

st.title("Customer Churn Prediction")
st.write("Predict customer churn risk using machine learning.")

try:
    bundle = load_model()
    model = bundle["model"]
    features = bundle["feature_columns"]
    threshold = bundle["threshold"]

    st.success("Model loaded successfully!")

    st.subheader("Customer Information")

    age = st.number_input("Age", 18, 65, 30)
    tenure = st.number_input("Tenure (months)", 1, 60, 12)
    usage = st.number_input("Usage Frequency", 1, 30, 15)
    last_interaction = st.number_input("Last Interaction (days)", 1, 30, 10)

    gender = st.selectbox("Gender", ["Male", "Female"])
    subscription = st.selectbox(
        "Subscription Type",
        ["Basic", "Premium", "Standard"]
    )

    if st.button("Predict Churn Risk", type="primary"):

        customer = pd.DataFrame([{
            "Age": age,
            "Tenure": tenure,
            "Usage Frequency": usage,
            "Last Interaction": last_interaction,
            "Usage_Intensity": usage / tenure,
            "Inactivity_Flag": int(last_interaction >= 20),
            "Gender_Male": int(gender == "Male"),
            "Subscription Type_Premium": int(subscription == "Premium"),
            "Subscription Type_Standard": int(subscription == "Standard")
        }])

        customer = customer.reindex(columns=features, fill_value=0)

        probability = model.predict_proba(customer)[0][1]
        prediction = int(probability >= threshold)

        st.subheader("Prediction Results")
        st.metric("Churn Probability", f"{probability * 100:.2f}%")

        if prediction:
            st.error("HIGHER CHURN RISK")
        else:
            st.success("LOWER CHURN RISK")

        st.caption(f"Classification threshold: {threshold:.2f}")

except Exception as e:
    st.error(f"Application error: {e}")

st.divider()
st.caption("Customer Churn Prediction | Data Science Internship")
