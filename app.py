import streamlit as st
import pandas as pd
import os
import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Student Performance Predictor", layout="wide")

st.title("📘 Student Math Score Prediction")
st.write("Upload dataset, train model, and predict student math scores.")

# -------------------------------
# Dataset Upload & Training
# -------------------------------
st.header("🔹 Train Model")

uploaded_file = st.file_uploader("Upload your dataset (stud.csv)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("✅ Dataset Preview", df.head())

        if st.button("Start Training"):
            # Save file temporarily for pipeline
            temp_path = "notebook/data/stud.csv"
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            df.to_csv(temp_path, index=False)

            # Run full pipeline
            obj = DataIngestion()
            train_data, test_data = obj.initiate_data_ingestion()

            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

            model_trainer = ModelTrainer()
            score = model_trainer.initiate_model_trainer(train_arr, test_arr)

            st.success(f"🎯 Model trained successfully! R2 Score: {score:.4f}")
    except Exception as e:
        st.error(f"Error during training: {str(e)}")

# -------------------------------
# Prediction Section
# -------------------------------
st.header("🔹 Predict Math Score")

with st.form("prediction_form"):
    gender = st.selectbox("Gender", ["male", "female"])
    race_ethnicity = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
    parental_level_of_education = st.selectbox(
        "Parental Level of Education",
        ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
    )
    lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
    test_preparation_course = st.selectbox("Test Preparation Course", ["none", "completed"])
    reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=50)
    writing_score = st.number_input("Writing Score", min_value=0, max_value=100, value=50)

    submitted = st.form_submit_button("Predict Score")

if submitted:
    try:
        custom_data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score,
        )
        input_df = custom_data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(input_df)

        st.success(f"📊 Predicted Math Score: **{prediction[0]:.2f}**")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
