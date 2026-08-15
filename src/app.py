import time
import streamlit as st
from src.image_utils import load_image_from_bytes
from src.predict import predict
from src.transforms import get_transform
from src.config import IMAGE_SIZE

st.set_page_config(page_title="Fabric Defect Detector", page_icon="🧵", layout="wide",)

# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .main-subtitle {
        text-align: center;
        color: #888;
        margin-top: 4px;
        margin-bottom: 18px;
    }

    .upload-box {
        border: 2px dashed #4a90e2;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 12px;
        text-align: center;
    }

    .stage-code {
        background-color: rgba(128, 128, 128, 0.10);
        border-radius: 10px;
        padding: 10px;
    }

    .stage-title {
        font-weight: 600;
        font-size: 1.05rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🧵 Fabric Defect Detector</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-subtitle">'
    'Fabric material and defect classification using ResNet'
    '</div>',
    unsafe_allow_html=True,
)

# ============================================================
# CONTROLS
# ============================================================

control_col1, control_col2, control_col3 = st.columns([1.2, 2.5, 1.5])

with control_col1:

    model_name = st.selectbox("🧠 Model", ["ResNet18", "ResNet50"],)

with control_col2:

    uploaded_file = st.file_uploader("📤 Upload fabric image", type=["jpg", "jpeg", "png", "bmp", "webp",], help="Upload an image of fabric to analyze.",)


with control_col3:

    st.write("")

    analyze = st.button("🔍 Analyze Image", use_container_width=True, type="primary",)

# ============================================================
# PIPELINE DEFINITIONS
# ============================================================

PIPELINE = [
    ("prepare_image", "📷", "Input / Preprocessing"),
    ("load_model", "🧠", "Load Model"),
    ("resnet", "⚡", "ResNet Feature Extraction"),
    ("fabric", "🧵", "Fabric Classification"),
    ("defect", "🔧", "Defect Classification"),
    ("prediction", "🎯", "Final Prediction"),
]

STAGE_CODE = {

    "prepare_image": """image = image.to(device)

if image.dim() == 3:
    image = image.unsqueeze(0)""",

    "load_model": """model = FabricDefectClassifier(model_name)

state_dict = torch.load(
    MODEL_PATHS[model_name],
    map_location=device
)

model.load_state_dict(state_dict)
model.eval()""",

    "resnet": """features = model.model(image)""",

    "fabric": """fabric_output = model.fabric_classifier(features)

fabric_probabilities = F.softmax(
    fabric_output,
    dim=1
)

fabric_confidence, fabric_index = torch.max(
    fabric_probabilities,
    dim=1
)""",

    "defect": """defect_output = model.defect_classifier(features)

defect_probabilities = F.softmax(
    defect_output,
    dim=1
)

defect_confidence, defect_index = torch.max(
    defect_probabilities,
    dim=1
)""",

    "prediction": """fabric_index = fabric_index.item()
defect_index = defect_index.item()

fabric = FABRIC_CLASSES[fabric_index]
defect = DEFECT_CLASSES[defect_index]""",
}

# ============================================================
# PIPELINE UI
# ============================================================

pipeline_placeholder = st.empty()

def render_pipeline(current_stage=None, completed=None):

    if completed is None:
        completed = set()

    with pipeline_placeholder.container():

        st.markdown("### 🔄 ML Pipeline")

        cols = st.columns(len(PIPELINE))

        for col, (stage_id, icon, title) in zip(cols, PIPELINE):

            with col:

                if stage_id in completed:
                    st.success(f"{icon} {title}\n\n✓ Done")

                elif stage_id == current_stage:
                    st.warning(f"{icon} {title}\n\n⏳ Running...")

                else:
                    st.info(f"{icon} {title}\n\nWaiting")

# ============================================================
# INITIAL PIPELINE
# ============================================================

render_pipeline()

# ============================================================
# MAIN WORKSPACE
# ============================================================

if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()

    workspace_left, workspace_right = st.columns([1.05, 1.4], gap="large",)

    # ========================================================
    # IMAGE
    # ========================================================

    with workspace_left:

        st.markdown("### 📷 Input Image")
        st.image(image_bytes, use_container_width=True,)
        st.caption(f"File: {uploaded_file.name}")

    # ========================================================
    # PROCESS + RESULTS
    # ========================================================

    with workspace_right:

        process_placeholder = st.empty()
        result_placeholder = st.empty()

        if analyze:

            completed_stages = set()

            transform = get_transform(augmentation=False)

            # ------------------------------------------------
            # CALLBACK
            # ------------------------------------------------

            def update_stage(stage_id, message):

                completed_stages.add(stage_id)

                render_pipeline(current_stage=stage_id, completed=completed_stages - {stage_id},)

                with process_placeholder.container():

                    st.markdown("### 💻 Current Process")
                    st.info(message)
                    st.code(STAGE_CODE.get(stage_id, "# Processing..."), language="python",)

                # Small delay makes the stages visible
                # without creating an artificial long animation.
                time.sleep(0.15)

            # ------------------------------------------------
            # PREPARE IMAGE
            # ------------------------------------------------

            update_stage("prepare_image", "Preparing uploaded image...",)

            image = load_image_from_bytes(image_bytes, image_size=IMAGE_SIZE, transform=transform,)

            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            results = predict(image, model_name.lower(), progress_callback=update_stage,)

            # ------------------------------------------------
            # FINISHED
            # ------------------------------------------------

            completed_stages = {
                stage_id
                for stage_id, _, _ in PIPELINE
            }

            render_pipeline(completed=completed_stages)

            with process_placeholder.container():

                st.markdown("### 💻 Current Process")
                st.success("✓ Pipeline completed successfully")

            # =================================================
            # RESULTS
            # =================================================

            with result_placeholder.container():

                st.markdown("### 🎯 Results")

                fabric_col, defect_col = st.columns(2)

                # ---------------------------------------------
                # FABRIC
                # ---------------------------------------------

                with fabric_col:

                    st.markdown("#### 🧵 Fabric")
                    st.success(results["fabric"])
                    st.metric("Confidence", f"{results['fabric_confidence'] * 100:.2f}%",)
                    st.progress(results["fabric_confidence"])

                # ---------------------------------------------
                # DEFECT
                # ---------------------------------------------

                with defect_col:

                    st.markdown("#### 🔧 Defect")
                    st.warning(results["defect"])
                    st.metric("Confidence", f"{results['defect_confidence'] * 100:.2f}%",)
                    st.progress(results["defect_confidence"])

                st.divider()

                # =================================================
                # PROBABILITIES
                # =================================================

                probability_col1, probability_col2 = st.columns(2)

                with probability_col1:

                    st.markdown("#### 🧵 Top Fabric Probabilities")

                    fabric_probabilities = (results["fabric_probabilities"])

                    top_fabrics = sorted(fabric_probabilities.items(), key=lambda item: item[1], reverse=True,)[:5]

                    for class_name, probability in top_fabrics:

                        st.write(
                            f"**{class_name}** — "
                            f"{probability * 100:.2f}%"
                        )

                        st.progress(probability)

                with probability_col2:

                    st.markdown("#### 🔧 Top Defect Probabilities")

                    defect_probabilities = (results["defect_probabilities"])

                    top_defects = sorted(defect_probabilities.items(), key=lambda item: item[1], reverse=True,)[:5]

                    for class_name, probability in top_defects:

                        st.write(
                            f"**{class_name}** — "
                            f"{probability * 100:.2f}%"
                        )
                        st.progress(probability)
                st.divider()
                st.caption(f"Model: {model_name}")
                st.caption("Task: Fabric Material and Defect Classification")
                st.caption("Fabric classes: 11 | Defect classes: 11")

        else:

            with process_placeholder.container():

                st.markdown("### 💻 Current Process")

                st.info(
                    "Upload an image and click "
                    "**Analyze Image** to start the pipeline."
                )

else:
    
    st.info("👆 Upload a fabric image above to start.")