import streamlit as st
from image_utils import load_image_from_bytes
from predict import predict
from transforms import get_transform
from config import IMAGE_SIZE

st.set_page_config(
    page_title="ML Toy Detector",
    page_icon="🧸",
    layout="wide"
)

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        margin-bottom: 5px;
    }

    .main-subtitle {
        text-align: center;
        margin-bottom: 35px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<h1 class="main-title">🧸 ML Toy Detector</h1>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="main-subtitle">'
    'Teddy Bear Defect Classification'
    '</p>',
    unsafe_allow_html=True
)

_, center, _ = st.columns([1, 2, 1])

with center:
    model_name = st.selectbox(
        "Select Model",
        ["ResNet18", "ResNet50"]
    )

_, center, _ = st.columns([1, 2, 1])

with center:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

_, center, _ = st.columns([1, 2, 1])

with center:
    analyze = st.button(
        "🔍 Analyze Image",
        use_container_width=True
    )

transform = get_transform(augmentation=False)

if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.image(
            image_bytes,
            caption="Uploaded Image",
            use_container_width=True
        )

    with col2:

        with st.container(border=True):

            st.subheader("🔍 Analysis Result")

            if analyze:

                image = load_image_from_bytes(image_bytes, image_size=IMAGE_SIZE, transform=transform)

                name, confidence = predict(image, model_name.lower())

                st.success(f"Prediction: {name}")

                st.metric("Confidence", f"{confidence * 100:.2f}%")

                st.progress(confidence)

                st.caption(f"Model: {model_name}")

                st.caption("Task: Teddy Bear Classification")

                st.caption("Classes: Normal / Defective")

            else:
                st.info(
                    "Click **Analyze Image** "
                    "to get a prediction."
                )

else:
    st.info("Please upload an image to get predictions.")