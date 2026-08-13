import torch
from model import TeddyClassifier
from config import (MODEL_RESNET18_PATH, MODEL_RESNET50_PATH, CLASSES,)

MODEL_PATHS = {
    "resnet18": MODEL_RESNET18_PATH,
    "resnet50": MODEL_RESNET50_PATH,
}

def predict(image, model_name):
    
    model_name = model_name.lower()

    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = TeddyClassifier(model_name)

    model.load_state_dict(torch.load(MODEL_PATHS[model_name], map_location="cpu"))

    model.eval()

    image = image.unsqueeze(0)

    with torch.no_grad():
        prediction = model(image)
        probabilities = torch.softmax(prediction, dim=1)
        predicted = prediction.argmax(dim=1)   
        name = CLASSES[predicted.item()]
        confidence = probabilities[0, predicted.item()].item()
        class_probabilities = {
            CLASSES[i]: probabilities[i].item()
            for i in range(len(CLASSES))
        } 
        return (name, confidence, class_probabilities)