import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Bank Customer Churn Prediction")
st.write("Predict whether a bank customer is likely to leave.")

@st.cache_resource
def load_model():
    return joblib.load("bank_churn_model.joblib")

try:
    model = load_model()
    st.success("Machine learning model loaded successfully!")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Get the original input columns from your trained pipeline
try:
    preprocessor = model.named_steps["preprocessor"]
    feature_columns = list(preprocessor.feature_names_in_)
except Exception:
    st.error(
        "Could not identify model input columns. "
        "Please check that the saved file contains your trained pipeline."
    )
    st.stop()

st.subheader("Enter Customer Details")
st.caption("Enter the customer's information below.")

numeric_defaults = {
    "RowNumber": 1,
    "CreditScore": 650,
    "Age": 40,
    "Tenure": 5,
    "Balance": 50000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 100000.0,
}

integer_features = {
    "RowNumber", "CreditScore", "Age", "Tenure",
    "NumOfProducts", "HasCrCard", "IsActiveMember"
}

categorical_options = {
    "Geography": ["France", "Germany", "Spain"],
    "Gender": ["Male", "Female"],
}

with st.form("customer_form"):
    values = {}

    col1, col2 = st.columns(2)

    for i, feature in enumerate(feature_columns):
        container = col1 if i % 2 == 0 else col2

        with container:
            if feature in categorical_options:
                values[feature] = st.selectbox(
                    feature,
                    categorical_options[feature]
                )
            else:
                default = numeric_defaults.get(feature, 0)

                if feature in integer_features:
                    values[feature] = st.number_input(
                        feature,
                        value=int(default),
                        step=1
                    )
                else:
                    values[feature] = st.number_input(
                        feature,
                        value=float(default),
                        step=1000.0 if feature == "EstimatedSalary"
                        else 100.0 if feature == "Balance"
                        else 1.0
                    )

    submitted = st.form_submit_button(
        "🔍 Predict Churn",
        type="primary",
        use_container_width=True
    )

if submitted:
    customer = pd.DataFrame(
        [[values[col] for col in feature_columns]],
        columns=feature_columns
    )

    try:
        prediction = model.predict(customer)[0]

        st.divider()
        st.subheader("Prediction Result")

        if int(prediction) == 1:
            st.error("⚠️ This customer is predicted to leave the bank.")
        else:
            st.success("✅ This customer is predicted to stay with the bank.")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(customer)[0]
            classes = list(model.classes_)

            if 1 in classes:
                churn_probability = float(
                    probabilities[classes.index(1)]
                )

                st.metric(
                    "Predicted Churn Probability",
                    f"{churn_probability * 100:.2f}%"
                )

                st.progress(churn_probability)

        with st.expander("View submitted customer data"):
            st.dataframe(customer, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Check that the model and scikit-learn version "
            "match the training environment."
        )