import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

from data_preprocessing import preprocess_data
from model import model, X_test, y_test

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")

# ==================================================
# Sidebar - Settings & CSV Upload
# ==================================================
st.sidebar.header("⚙️ Settings")

uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Custom dataset uploaded!")
else:
    data = pd.read_csv("data/churn_data.csv")
    st.sidebar.info("ℹ️ Using default dataset (Telco Churn).")

# ==================================================
# Tabs Layout
# ==================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📑 Dataset", "📈 Model Performance", "🔍 Feature Importance", "🔮 Prediction"]
)

# ==================================================
# 1. Dataset Tab
# ==================================================
with tab1:
    st.subheader("📑 Dataset Overview")
    st.dataframe(data.head())

    st.subheader("Churn Distribution")
    fig = px.histogram(data, x="Churn", title="Churn Distribution")
    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# 2. Model Performance Tab
# ==================================================
with tab2:
    st.subheader("📈 Model Accuracy & Confusion Matrix")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.metric("Model Accuracy", f"{acc*100:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# ==================================================
# 3. Feature Importance Tab
# ==================================================
with tab3:
    st.subheader("🔍 Feature Importance")

    feature_importances = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig = px.bar(
        feature_importances,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance"
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# 4. Prediction Tab
# ==================================================
with tab4:
    st.subheader("🔮 Predict Churn for a Customer")

    customer_features = {}
    for col in X_test.columns:
        customer_features[col] = st.number_input(f"Enter {col}", value=0)

    if st.button("Predict"):
        input_df = pd.DataFrame([customer_features])
        prediction = model.predict(input_df)

        result_df = input_df.copy()
        result_df["Churn_Prediction"] = ["Yes" if prediction[0] == 1 else "No"]

        st.success("Churn Prediction: **Yes**" if prediction[0] == 1 else "Churn Prediction: **No**")

        # Download button
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Prediction Result as CSV",
            data=csv,
            file_name="prediction_result.csv",
            mime="text/csv",
        )
