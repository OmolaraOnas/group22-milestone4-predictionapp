import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

# Initialize global variables
df = None
model_pipeline = None

# App title
st.title("Milestone 4 Regression Application")

# Upload File Component
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset preview:")
    st.dataframe(df)

# Select Target Component
if df is not None:
    target = st.selectbox("Select Target Variable", df.select_dtypes(include=np.number).columns)

# Bar Chart Components
if df is not None and target:
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

    # First Chart: Average of Target by Category
    selected_category = st.selectbox("Select a Categorical Variable for Analysis", categorical_cols)
    if selected_category:
        avg_data = df.groupby(selected_category)[target].mean().reset_index()
        fig1 = px.bar(avg_data, x=selected_category, y=target, title=f"Average {target} by {selected_category}")
        st.plotly_chart(fig1)

    # Second Chart: Correlation Strength
    corr_data = df[numerical_cols].corr()[target].abs().reset_index()
    corr_data.columns = ['Feature', 'Correlation']
    fig2 = px.bar(corr_data, x='Feature', y='Correlation', title=f"Correlation with {target}")
    st.plotly_chart(fig2)

# Train Component
if df is not None:
    st.header("Train Model")
    features = st.multiselect("Select Features for Training", numerical_cols)
    if st.button("Train Model"):
        if not features:
            st.warning("Please select at least one feature for training.")
        else:
            X = df[features]
            y = df[target]

            # Preprocessing pipeline
            num_imputer = SimpleImputer(strategy='mean')
            cat_imputer = SimpleImputer(strategy='most_frequent')
            num_cols = X.select_dtypes(include=["number"]).columns
            cat_cols = X.select_dtypes(include=["object"]).columns

            preprocessor = ColumnTransformer([
                ('num', Pipeline([('imputer', num_imputer), ('scaler', StandardScaler())]), num_cols),
                ('cat', Pipeline([('imputer', cat_imputer), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
            ])

            model_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', LinearRegression())
            ])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model_pipeline.fit(X_train, y_train)
            y_pred = model_pipeline.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            st.success(f"Model trained successfully! R² score: {r2:.2f}")

# Prediction Component
if model_pipeline is not None:
    st.header("Make Predictions")
    input_values = st.text_input("Enter feature values (comma-separated)")
    if st.button("Predict"):
        try:
            input_list = [float(x) for x in input_values.split(',')]
            if len(input_list) != len(features):
                st.error(f"Expected {len(features)} values, but got {len(input_list)}.")
            else:
                input_df = pd.DataFrame([input_list], columns=features)
                prediction = model_pipeline.predict(input_df)[0]
                st.success(f"Predicted value: {prediction:.2f}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")
