import torch

from model import TeddyClassifier
from config import MODEL_PATH, CLASSES,IMAGE_SIZE
from transforms import get_transform
from image_utils import load_image

model = TeddyClassifier()

transform = get_transform(augmentation=False)
    
image = load_image("test_image.jpg", image_size=IMAGE_SIZE, transform=transform)

model.load_state_dict(torch.load(MODEL_PATH))

model.eval()

image = image.unsqueeze(0)

with torch.no_grad():
    
    prediction = model(image)
    
    predicted = prediction.argmax(dim=1)

    probabilities = torch.softmax(prediction, dim=1)

    confidence = probabilities[0, predicted.item()].item()
    
    print(f"Confidence: {confidence * 100:.4f}%")
    print(f"Predicted: {CLASSES[predicted.item()]}")