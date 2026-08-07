import os

from image_utils import load_image
from torch.utils.data import Dataset
from transforms import get_transform
from config import CLASSES

class TeddyDataset(Dataset):

    def __init__(self, dataset_path, image_size, augmentation):

        self.dataset_path = dataset_path
        self.image_size = image_size
        self.augmentation = augmentation
        
        self.images = []
        self.transform = get_transform(augmentation=self.augmentation)

        for label, class_name in enumerate(CLASSES):
            class_path = os.path.join(self.dataset_path, class_name)
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                self.images.append((image_path, label))
           
    def __len__(self):
        return len(self.images)
         
    def __getitem__(self, index):
        path, label = self.images[index]
        image = load_image(path, self.image_size, self.transform, self.augmentation)
        return image, label