import streamlit as st
from image_utils import load_image_from_bytes
from predict import predict
from transforms import get_transform
from config import IMAGE_SIZE

st.set_page_config(page_title="Fabric Defect Detector", page_icon="🧵", layout="wide")

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
    '<h1 class="main-title">🧵 Fabric Defect Detector</h1>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="main-subtitle">'
    'Fabric Material and Defect Classification'
    '</p>',
    unsafe_allow_html=True
)

# ============================================================
# MODEL SELECTION
# ============================================================

_, center, _ = st.columns([1, 2, 1])

with center:

    model_name = st.selectbox("Select Model", ["ResNet18", "ResNet50"])

# ============================================================
# IMAGE UPLOAD
# ============================================================

_, center, _ = st.columns([1, 2, 1])

with center:

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

# ============================================================
# ANALYZE BUTTON
# ============================================================

_, center, _ = st.columns([1, 2, 1])

with center:

    analyze = st.button("🔍 Analyze Image", use_container_width=True)

# ============================================================
# ML PIPELINE
# ============================================================

if uploaded_file is not None:

    st.subheader("🔄 ML Pipeline")

    pipeline = [
        "📷 Input Image",
        "🔄 Resize",
        "🔢 ToTensor",
        "📊 Normalize",
        "🧠 ResNet",
        "🧵 Fabric",
        "🔧 Defect",
        "🎯 Prediction",
    ]

    cols = st.columns(len(pipeline))

    for col, step in zip(cols, pipeline):

        with col:
            st.info(step)


# ============================================================
# IMAGE + RESULT
# ============================================================

if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()

    transform = get_transform(augmentation=False)

    col1, col2 = st.columns(2, gap="large")

    # ========================================================
    # INPUT IMAGE
    # ========================================================

    with col1:

        st.subheader("📷 Input Image")
        st.image(image_bytes, use_container_width=True)

    # ========================================================
    # ANALYSIS RESULT
    # ========================================================

    with col2:

        with st.container(border=True):

            st.subheader("🔍 Analysis Result")

            if analyze:

                image = load_image_from_bytes(image_bytes, image_size=IMAGE_SIZE, transform=transform)
                results = predict(image, model_name.lower())

                # =================================================
                # FABRIC RESULT
                # =================================================

                st.subheader("🧵 Fabric")
                st.success(f"Prediction: {results['fabric']}")
                st.metric("Confidence", f"{results['fabric_confidence'] * 100:.2f}%")
                st.progress(results["fabric_confidence"])

                # =================================================
                # DEFECT RESULT
                # =================================================

                st.subheader("🔧 Defect")
                st.warning(f"Prediction: {results['defect']}")
                st.metric("Confidence", f"{results['defect_confidence'] * 100:.2f}%")
                st.progress(results["defect_confidence"])

                # =================================================
                # FABRIC PROBABILITIES
                # =================================================

                st.divider()
                st.subheader("🧵 Fabric Class Probabilities")
                fabric_probabilities = results["fabric_probabilities"]

                for class_name, probability in sorted(fabric_probabilities.items(), key=lambda item: item[1], reverse=True)[:5]:
                    st.write(
                        f"{class_name}: "
                        f"{probability * 100:.2f}%"
                    )
                    st.progress(probability)


                # =================================================
                # DEFECT PROBABILITIES
                # =================================================

                st.divider()
                st.subheader("🔧 Defect Class Probabilities")
                defect_probabilities = results["defect_probabilities"]

                for class_name, probability in sorted(defect_probabilities.items(), key=lambda item: item[1], reverse=True)[:5]:

                    st.write(
                        f"{class_name}: "
                        f"{probability * 100:.2f}%"
                    )
                    st.progress(probability)

                # =================================================
                # MODEL INFORMATION
                # =================================================

                st.divider()
                st.caption(f"Model: {model_name}")
                st.caption("Task: Fabric Material and Defect Classification")
                st.caption("Fabric Classes: 11")
                st.caption("Defect Classes: 11")


            else:
                st.info(
                    "Click **Analyze Image** "
                    "to get predictions."
                )

else:
    st.info("Please upload an image to get predictions.")