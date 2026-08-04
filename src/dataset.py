import os
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

class TeddyDataset(Dataset):

    def __init__(self, dataset_path, image_size, augmentation):

        self.dataset_path = dataset_path
        self.image_size = image_size
        self.augmentation = augmentation
        
        self.images = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        classes = ("normal", "defective")

        for label, class_name in enumerate(classes):
            class_path = os.path.join(self.dataset_path, class_name)
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                self.images.append((image_path, label))
           
    def __len__(self):
        return len(self.images)
         
    def __getitem__(self, index):
        path, label = self.images[index]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        if self.augmentation:
            image = self.augmentation(image=image)["image"]
        image = self.transform(image)
        return image, label
    