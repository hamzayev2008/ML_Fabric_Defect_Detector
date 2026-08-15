import torch
import torch.nn.functional as F
from config import (
    MODEL_RESNET18_PATH,
    MODEL_RESNET50_PATH,
    FABRIC_CLASSES,
    DEFECT_CLASSES,
)
from model import FabricDefectClassifier

MODEL_PATHS = {
    "resnet18": MODEL_RESNET18_PATH,
    "resnet50": MODEL_RESNET50_PATH,
}

def load_model(model_name, device):

    model_name = model_name.lower()

    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unknown model: {model_name}")

    model = FabricDefectClassifier(model_name)

    state_dict = torch.load(MODEL_PATHS[model_name], map_location=device)

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model

def predict(image, model_name):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_name, device)

    image = image.to(device)

    if image.dim() == 3:
        image = image.unsqueeze(0)

    with torch.no_grad():

        fabric_output, defect_output = model(image)

        fabric_probabilities = F.softmax(fabric_output, dim=1)

        defect_probabilities = F.softmax(defect_output, dim=1)

        fabric_confidence, fabric_index = torch.max(fabric_probabilities, dim=1)

        defect_confidence, defect_index = torch.max(defect_probabilities, dim=1)

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