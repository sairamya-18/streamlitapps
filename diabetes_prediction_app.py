import streamlit as st
from joblib import load
import numpy as np
from PIL import Image

# Load the trained model
model_path = r"C:\Users\91998\OneDrive\Documents\diabetes_prediction_model.joblib"  # Specify the path to your model
model = load(model_path)

# Title of the application
st.title('Diabetes Prediction App')

# Description of the application
st.write('This app predicts the likelihood of diabetes based on input features.')

# Sidebar with input fields
st.sidebar.header('Input Features')

# Input fields for user to input data
pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 1)
glucose = st.sidebar.slider('Glucose (mg/dL)', 0, 200, 100)
blood_pressure = st.sidebar.slider('Blood Pressure (mm Hg)', 0, 122, 70)
skin_thickness = st.sidebar.slider('Skin Thickness (mm)', 0, 99, 20)
insulin = st.sidebar.slider('Insulin (mu U/ml)', 0, 846, 80)
bmi = st.sidebar.slider('BMI', 0.0, 67.1, 30.0)
dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725)
age = st.sidebar.slider('Age (years)', 21, 81, 30)

# Convert input data to numpy array
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
doctor_image = Image.open(r"C:\Users\91998\OneDrive\Documents\doctor_img.jpeg")  # Path to your doctor image

# Display doctor image
st.image(doctor_image, caption='Dr.Diabetes Predicter')
# Button to make predictions
if st.sidebar.button('Predict'):
    # Predict diabetes likelihood
    prediction = model.predict(input_data)
    
    # Display prediction result
    if prediction[0] == 0:
        st.success('Based on the input features, the individual is predicted to NOT have diabetes.')
    else:
        st.error('Based on the input features, the individual is predicted to have diabetes.')
