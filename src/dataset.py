import os
import cv2
from torch.utils.data import Dataset

class TeddyDataset(Dataset):

    def __init__(self, dataset_path, image_size, augmentation):

        self.dataset_path = dataset_path
        self.image_size = image_size
        self.augmentation = augmentation

        self.images = []
        classes = ("normal", "defective")

        for label, class_name in enumerate(classes):
            class_path = os.path.join(self.dataset_path, class_name)
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                self.images.append((image_path, label))