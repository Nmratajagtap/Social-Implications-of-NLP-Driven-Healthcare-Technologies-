import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Title
st.title("Customer Churn Analysis App")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Data Info
    st.subheader("Dataset Info")
    st.write(df.describe())
    st.write("Shape:", df.shape)

    # Handle Categorical Features
    label_encoders = {}
    for column in df.select_dtypes(include=["object"]).columns:
        if column != 'Churn':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

    # Encode target
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # EDA - Churn count
    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='Churn', ax=ax)
    st.pyplot(fig)

    # Feature Selection
    target = 'Churn'
    X = df.drop(target, axis=1)
    y = df[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    st.subheader("Choose Model")
    model_name = st.selectbox("Select Model", ["Logistic Regression", "Decision Tree", "Random Forest"])

    if st.button("Train Model"):
        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_name == "Random Forest":
            model = RandomForestClassifier()

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        acc = accuracy_score(y_test, predictions)
        st.success(f"Model Accuracy: {acc:.2f}")

        st.subheader("Classification Report")
        st.text(classification_report(y_test, predictions))
