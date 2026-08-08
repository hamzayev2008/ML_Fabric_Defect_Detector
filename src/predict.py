import torch

from model import TeddyClassifier
from config import MODEL_PATH, CLASSES,IMAGE_SIZE
from transforms import get_transform
from image_utils import load_image

model = TeddyClassifier()

transform = get_transform(augmentation=False)

model.load_state_dict(torch.load(MODEL_PATH))

model.eval()

def predict(image_path):
    image = load_image(image_path, image_size=IMAGE_SIZE, transform=transform)
    image = image.unsqueeze(0)

    with torch.no_grad():
        prediction = model(image)
        predicted = prediction.argmax(dim=1)
        probabilities = torch.softmax(prediction, dim=1)
        confidence = probabilities[0, predicted.item()].item()
        
        name = CLASSES[predicted.item()]
        
        return name, confidence