import torch
import torch.nn.functional as F

from src.config import (
    MODEL_RESNET18_PATH,
    MODEL_RESNET50_PATH,
    FABRIC_CLASSES,
    DEFECT_CLASSES,
)

from src.model import FabricDefectClassifier
from src.domain_model import DomainClassifier


MODEL_PATHS = {
    "resnet18": MODEL_RESNET18_PATH,
    "resnet50": MODEL_RESNET50_PATH,
}

DOMAIN_MODEL_PATH = "domain_gate_resnet18_v2.pth"


def load_model(model_name, device, progress_callback=None):

    model_name = model_name.lower()

    if model_name not in MODEL_PATHS:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    if progress_callback:
        progress_callback(
            "load_model",
            "Loading main model..."
        )

    model = FabricDefectClassifier(model_name)

    state_dict = torch.load(
        MODEL_PATHS[model_name],
        map_location=device
    )

    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model


def load_domain_model(device, progress_callback=None):

    if progress_callback:
        progress_callback(
            "domain_gate",
            "Checking whether the image belongs to the fabric domain..."
        )

    model = DomainClassifier().to(device)

    state_dict = torch.load(
        DOMAIN_MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model


def predict(image, model_name, progress_callback=None):

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    # ========================================================
    # DOMAIN GATE
    # ========================================================

    domain_model = load_domain_model(
        device,
        progress_callback
    )

    image = image.to(device)

    if image.dim() == 3:
        image = image.unsqueeze(0)

    with torch.no_grad():

        domain_output = domain_model(image)

        domain_probabilities = F.softmax(
            domain_output,
            dim=1
        )

        domain_confidence, domain_index = torch.max(
            domain_probabilities,
            dim=1
        )

    domain_index = domain_index.item()
    domain_confidence = domain_confidence.item()

    domain_prediction = (
        "Fabric"
        if domain_index == 0
        else "Non-Fabric"
    )

    # --------------------------------------------------------
    # REJECT NON-FABRIC
    # --------------------------------------------------------

    if domain_prediction == "Non-Fabric":

        if progress_callback:
            progress_callback(
                "domain_reject",
                "Image rejected: it does not appear to be fabric."
            )

        return {
            "valid_input": False,

            "domain": domain_prediction,
            "domain_confidence": domain_confidence,

            "fabric": None,
            "fabric_confidence": None,
            "fabric_probabilities": None,

            "defect": None,
            "defect_confidence": None,
            "defect_probabilities": None,
        }

    # ========================================================
    # LOAD MAIN MODEL
    # ========================================================

    model = load_model(
        model_name,
        device,
        progress_callback
    )

    # ========================================================
    # RESNET FEATURES
    # ========================================================

    if progress_callback:
        progress_callback(
            "resnet",
            "Running ResNet feature extraction..."
        )

    with torch.no_grad():

        features = model.model(image)

    # ========================================================
    # FABRIC CLASSIFICATION
    # ========================================================

    if progress_callback:
        progress_callback(
            "fabric",
            "Classifying fabric material..."
        )

    with torch.no_grad():

        fabric_output = (
            model.fabric_classifier(features)
        )

        fabric_probabilities = F.softmax(
            fabric_output,
            dim=1
        )

        fabric_confidence, fabric_index = torch.max(
            fabric_probabilities,
            dim=1
        )

    # ========================================================
    # DEFECT CLASSIFICATION
    # ========================================================

    if progress_callback:
        progress_callback(
            "defect",
            "Classifying fabric defect..."
        )

    with torch.no_grad():

        defect_output = (
            model.defect_classifier(features)
        )

        defect_probabilities = F.softmax(
            defect_output,
            dim=1
        )

        defect_confidence, defect_index = torch.max(
            defect_probabilities,
            dim=1
        )

    # ========================================================
    # FINAL PREDICTION
    # ========================================================

    if progress_callback:
        progress_callback(
            "prediction",
            "Preparing final prediction..."
        )

    fabric_index = fabric_index.item()
    defect_index = defect_index.item()

    fabric_confidence = (
        fabric_confidence.item()
    )

    defect_confidence = (
        defect_confidence.item()
    )

    fabric_probabilities = {
        FABRIC_CLASSES[i]:
        fabric_probabilities[0][i].item()
        for i in range(len(FABRIC_CLASSES))
    }

    defect_probabilities = {
        DEFECT_CLASSES[i]:
        defect_probabilities[0][i].item()
        for i in range(len(DEFECT_CLASSES))
    }

    return {
        "valid_input": True,

        "domain": domain_prediction,
        "domain_confidence": domain_confidence,

        "fabric": FABRIC_CLASSES[fabric_index],
        "fabric_confidence": fabric_confidence,
        "fabric_probabilities": fabric_probabilities,

        "defect": DEFECT_CLASSES[defect_index],
        "defect_confidence": defect_confidence,
        "defect_probabilities": defect_probabilities,
    }