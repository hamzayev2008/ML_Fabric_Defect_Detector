import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from config import MODEL_PATH, CLASSES
from model import TeddyClassifier
from dataLoader import test_loader

model = TeddyClassifier()

true_labels = []
predicted_labels = []

state_dict = torch.load(MODEL_PATH)
model.load_state_dict(state_dict)
model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        true_labels.extend(labels.tolist())
        predicted_labels.extend(predicted.tolist())

cm = confusion_matrix(true_labels, predicted_labels)

print("Confusion Matrix:")
print(cm)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.xticks(range(len(CLASSES)), CLASSES)
plt.yticks(range(len(CLASSES)), CLASSES)
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')
plt.show()