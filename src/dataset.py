from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from transforms import get_transform

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class FabricDataset(Dataset):

    def __init__(self, dataset_path, augmentation=False):
        
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        self.transform = get_transform(augmentation=augmentation)

        self.fabric_classes = sorted([
            folder.name
            for folder in self.dataset_path.iterdir()
            if folder.is_dir()
        ])

        self.fabric_to_index = {
            name: index
            for index, name in enumerate(self.fabric_classes)
        }

        self.defect_classes = sorted({
            defect.name
            for fabric_folder in self.dataset_path.iterdir()
            if fabric_folder.is_dir()
            for defect in fabric_folder.iterdir()
            if defect.is_dir()
        })

        self.defect_to_index = {
            name: index
            for index, name in enumerate(self.defect_classes)
        }

        self.images = []

        for fabric_folder in self.dataset_path.iterdir():

            if not fabric_folder.is_dir():
                continue

            fabric_name = fabric_folder.name

            for defect_folder in fabric_folder.iterdir():

                if not defect_folder.is_dir():
                    continue

                defect_name = defect_folder.name

                for image_path in defect_folder.iterdir():

                    if (
                        image_path.is_file()
                        and image_path.suffix.lower()
                        in IMAGE_EXTENSIONS
                    ):
                        self.images.append((image_path, self.fabric_to_index[fabric_name], self.defect_to_index[defect_name]))

        if not self.images:
            raise ValueError(f"No images found in {self.dataset_path}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_path, fabric_label, defect_label = self.images[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return (image, fabric_label, defect_label)