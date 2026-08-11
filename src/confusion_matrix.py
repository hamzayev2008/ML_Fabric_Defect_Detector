import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from config import MODEL_PATH, CLASSES, BATCH_SIZE
from model import TeddyClassifier
from dataLoader import test_loader

model = TeddyClassifier()

true_labels = []
predicted_labels = []
wrong_images = []

state_dict = torch.load(MODEL_PATH)
model.load_state_dict(state_dict)
model.eval()

batch_start = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        true_labels.extend(labels.tolist())
        predicted_labels.extend(predicted.tolist())
        wrong = predicted != labels
        wrong_indices = wrong.nonzero().flatten().tolist()
        for wrong_index in wrong_indices:
            global_index = batch_start + wrong_index
            path, actual_label = test_loader.dataset.images[global_index]
            predicted_label = predicted[wrong_index]
            wrong_images.append((path, actual_label, predicted_label))
        batch_start += BATCH_SIZE
        
print(wrong_images)
        
report = classification_report(true_labels, predicted_labels)
print(report)

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

plt.subplot(1, 2, 1)
plt.title(wrong_images[0][1].item(CLASSES[labels.item()]))
plt.title(wrong_images[0][2].item(CLASSES[predicted.item()]))
image = plt.imread(wrong_images[0][0])
plt.imshow(image)
plt.subplot(1, 2, 2)
plt.title(wrong_images[1][1].item(CLASSES[labels.item()]))
plt.title(wrong_images[1][2].item(CLASSES[predicted.item()]))
plt.title(wrong_images[1])
image = plt.imread(wrong_images[1][0])
plt.imshow(image)
plt.show()