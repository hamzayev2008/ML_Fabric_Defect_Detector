import os
from sklearn.model_selection import train_test_split
from config import CLASSES, DATASET_PATH
import shutil

images = []
labels = []

for label, class_name in enumerate(CLASSES):
    class_path = os.path.join(DATASET_PATH, class_name)
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)
        images.append(image_path)
        labels.append(label)
                
train_images, temp_images, train_labels, temp_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

validate_images, test_images, validate_labels, test_labels = train_test_split(
    temp_images, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

os.makedirs("train_dataset", exist_ok=True)
for image_path, label in zip(train_images, train_labels):
    class_name = CLASSES[label]
    class_path = os.path.join("train_dataset", class_name)
    os.makedirs(class_path, exist_ok=True)
    shutil.copy(image_path, class_path)

os.makedirs("validation_dataset", exist_ok=True)
for image_path, label in zip(validate_images, validate_labels):
    class_name = CLASSES[label]
    class_path = os.path.join("validation_dataset", class_name)
    os.makedirs(class_path, exist_ok=True)
    shutil.copy(image_path, class_path)

os.makedirs("test_dataset", exist_ok=True)
for image_path, label in zip(test_images, test_labels):
    class_name = CLASSES[label]
    class_path = os.path.join("test_dataset", class_name)
    os.makedirs(class_path, exist_ok=True)
    shutil.copy(image_path, class_path)