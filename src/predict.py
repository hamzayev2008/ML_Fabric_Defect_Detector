import torch
import torch.nn.functional as F
from src.config import (
    MODEL_RESNET18_PATH,
    MODEL_RESNET50_PATH,
    FABRIC_CLASSES,
    DEFECT_CLASSES,
)
from src.model import FabricDefectClassifier

MODEL_PATHS = {
    "resnet18": MODEL_RESNET18_PATH,
    "resnet50": MODEL_RESNET50_PATH,
}

def load_model(model_name, device, progress_callback=None):

    model_name = model_name.lower()

    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unknown model: {model_name}")

    if progress_callback:
        progress_callback("load_model", "Loading model...")

    model = FabricDefectClassifier(model_name)

    state_dict = torch.load(MODEL_PATHS[model_name], map_location=device)

    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model

def predict(image, model_name, progress_callback=None):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = load_model(model_name, device, progress_callback)

    # ========================================================
    # PREPARE IMAGE
    # ========================================================

    if progress_callback:
        progress_callback("prepare_image", "Preparing image...")

    image = image.to(device)

    if image.dim() == 3:
        image = image.unsqueeze(0)

    # ========================================================
    # RESNET
    # ========================================================

    if progress_callback:
        progress_callback("resnet", "Running ResNet...")

    with torch.no_grad():

        features = model.model(image)

    # ========================================================
    # FABRIC CLASSIFICATION
    # ========================================================

    if progress_callback:
        progress_callback("fabric", "Classifying fabric...")

    with torch.no_grad():

        fabric_output = model.fabric_classifier(features)

        fabric_probabilities = F.softmax(fabric_output, dim=1)

        fabric_confidence, fabric_index = torch.max(fabric_probabilities, dim=1)

    # ========================================================
    # DEFECT CLASSIFICATION
    # ========================================================

    if progress_callback:
        progress_callback("defect", "Classifying defect...")

    with torch.no_grad():

        defect_output = model.defect_classifier(features)

        defect_probabilities = F.softmax(defect_output, dim=1)

        defect_confidence, defect_index = torch.max(defect_probabilities, dim=1)

    # ========================================================
    # PREDICTION
    # ========================================================

    if progress_callback:
        progress_callback("prediction", "Preparing prediction...")

    fabric_index = fabric_index.item()
    defect_index = defect_index.item()

    fabric_confidence = fabric_confidence.item()
    defect_confidence = defect_confidence.item()

    fabric_probabilities = {
        FABRIC_CLASSES[i]: fabric_probabilities[0][i].item()
        for i in range(len(FABRIC_CLASSES))
    }

    defect_probabilities = {
        DEFECT_CLASSES[i]: defect_probabilities[0][i].item()
        for i in range(len(DEFECT_CLASSES))
    }

    return {
        "fabric": FABRIC_CLASSES[fabric_index],
        "fabric_confidence": fabric_confidence,
        "fabric_probabilities": fabric_probabilities,

        "defect": DEFECT_CLASSES[defect_index],
        "defect_confidence": defect_confidence,
        "defect_probabilities": defect_probabilities,
    }