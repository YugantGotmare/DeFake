import streamlit as st
from PIL import Image
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

# Define IMAGE_SIZE here
IMAGE_SIZE = 128

# Load the model architecture from the JSON file
with open('fakevsreal_model.json', 'r') as json_file:
    json_savedModel = json_file.read()

# Load the model weights
model = model_from_json(json_savedModel)
model.load_weights('fakevsreal_weights.h5')
model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])

def read_and_preprocess(img, image_size):
    # Convert BytesIO to image
    img = Image.open(img).convert('RGB')
    
    # Resize and preprocess the image
    img = np.array(img.resize((image_size, image_size)))
    img = img / 255.0
    return img

def classify_image(image, image_size):
    # Preprocess the image
    processed_image = read_and_preprocess(image, image_size)
    
    # Reshape for model input
    processed_image = processed_image.reshape(1, image_size, image_size, 3)

    # Make prediction
    prediction = model.predict(processed_image)

    # Obtain the predicted class
    predicted_class = np.argmax(prediction)

    return predicted_class

# Streamlit web app
st.title('Fake vs Real Image Classifier')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Create a layout with two columns
    col1, col2 = st.columns(2)

    # Display the image in the first column
    col1.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

    # Classify the uploaded image
    result = classify_image(uploaded_file, IMAGE_SIZE)

    # Display the prediction in the second column with custom formatting
    labels = ['Real', 'Fake']
    prediction_text = f"**Prediction:** {labels[result]}"

    if result == 0:  # Real
        col2.markdown(f"<p style='font-size:24px;color:green;'>{prediction_text}</p>", unsafe_allow_html=True)
    else:  # Fake
        col2.markdown(f"<p style='font-size:24px;color:red;'>{prediction_text}</p>", unsafe_allow_html=True)
