

import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model_path = r"C:\Users\91998\OneDrive\Documents\model_name.h5"
model = load_model(model_path)

# Define function to preprocess and make predictions
def preprocess_image(image):
    # Resize the image to 48x48
    img_resized = image.resize((48, 48))
    # Convert the image to grayscale
    img_gray = img_resized.convert('L')
    # Convert the image to a numpy array
    img_array = np.array(img_gray)
    # Add a batch dimension and normalize pixel values
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_emotion(image):
    # Preprocess the image
    img_array = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(img_array)
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    emotion = emotion_labels[np.argmax(prediction)]
    
    return emotion

# Define the Streamlit UI
st.title('Emotion Recognition App')
st.write('Upload an image and we will predict the emotion!')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Make prediction when button is clicked
    if st.button('Predict'):
        emotion = predict_emotion(image)
        st.success(f'Predicted Emotion: {emotion}')
