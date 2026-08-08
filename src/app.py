import streamlit as st
from image_utils import load_image_from_bytes
from predict import predict
from transforms import get_transform
from config import IMAGE_SIZE

st.write("Teddy Detector")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
transform = get_transform(augmentation=False)

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    
    st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
    
    image = load_image_from_bytes(image_bytes, image_size=IMAGE_SIZE, transform=transform)
    name, confidence = predict(image)
    
    st.write(f"Predicted: {name}")
    st.write(f"Confidence: {confidence * 100:.2f}%")
    
else:
    st.write("Please upload an image to get predictions.")