import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the model
model = load_model('my_model.h5')

# Define class names
class_names = ['closed', 'opened']

# Streamlit UI
st.set_page_config(page_title="Sleepy Driver Detection", page_icon="🚗")
st.title("Sleepy Driver Detection App 🚗")
st.markdown("<h3 style='text-align: center;'>Live Camera Feed for Drowsiness Detection</h3>", unsafe_allow_html=True)

# OpenCV Video Capture
run = st.checkbox("Start Live Detection")

if run:
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    # Streamlit placeholder for displaying video
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture image")
            break
        frame = cv2.flip(frame, 1)  # Flip frame for natural interaction
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(frame_rgb, (180, 180))  # Resize for model input
        img_array = img_to_array(resized_frame)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict using the model
        prediction = model.predict(img_array)
        confidence = prediction[0]

        # Determine the predicted class
        if confidence > 0.5:
            predicted_class = class_names[1]  # 'opened'
            label = "Awake 😊"
            color = (0, 255, 0)  # Green
        else:
            predicted_class = class_names[0]  # 'closed'
            label = "Drowsy! ⚠️"
            color = (0, 0, 255)  # Red

        # Display result on the video frame
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Show the frame in Streamlit
        frame_placeholder.image(frame, channels="BGR", use_container_width=True)

    cap.release()
