import torch

from model import TeddyClassifier
from config import (MODEL_RESNET18_PATH, MODEL_RESNET50_PATH, CLASSES,)

MODEL_PATHS = {
    "resnet18": MODEL_RESNET18_PATH,
    "resnet50": MODEL_RESNET50_PATH,
}

def predict(image, model_name):
    
    model = TeddyClassifier(model_name)

    model.load_state_dict(torch.load(MODEL_PATHS[model_name], map_location="cpu"))

    model.eval()

    image = image.unsqueeze(0)

    with torch.no_grad():
        prediction = model(image)
        predicted = prediction.argmax(dim=1)
        probabilities = torch.softmax(prediction, dim=1)
        confidence = probabilities[0, predicted.item()].item()    
        name = CLASSES[predicted.item()]   
        return name, confidence