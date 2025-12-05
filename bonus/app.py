# bonus/app.py - MNIST Digit Classifier (FIXED PATH + NO ERROR)
# Run: python -m streamlit run bonus/app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os

# FIXED: Use correct path to model
model_path = os.path.join('bonus', 'mnist_model.h5')
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None

model = load_model()

st.title("Handwritten Digit Classifier")
st.write("Draw a digit (0–9) below:")

# Canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

if model and canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data).convert("L")
    img = Image.fromarray(255 - np.array(img))
    img = img.resize((28, 28))
    img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0

    pred = model.predict(img_array, verbose=0)
    digit = int(pred.argmax())
    confidence = float(pred.max())

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Your Drawing", width=150)
    with col2:
        st.subheader(f"**Prediction: {digit}**")
        st.metric("Confidence", f"{confidence:.1%}")

    st.bar_chart(pred[0])