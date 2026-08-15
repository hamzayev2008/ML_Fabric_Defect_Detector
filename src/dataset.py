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
        self.images = []
        fabric_folders = [
            folder
            for folder in self.dataset_path.iterdir()
            if folder.is_dir()
        ]
        fabric_folders.sort(key=lambda folder: folder.name)
        self.fabric_classes = [
            folder.name[3:]
            if len(folder.name) > 3
            else folder.name
            for folder in fabric_folders
        ]
        self.fabric_to_index = {
            name: index
            for index, name in enumerate(self.fabric_classes)
        }

        defect_names = set()

        for fabric_folder in fabric_folders:
            for defect_folder in fabric_folder.iterdir():
                if defect_folder.is_dir():
                    defect_names.add(defect_folder.name)

        defect_names = sorted(defect_names, key=lambda name: int(name.split(".")[0]))
        self.defect_classes = [
            name.split(". ", 1)[1]
            for name in defect_names
        ]
        self.defect_to_index = {
            name: index
            for index, name in enumerate(self.defect_classes)
        }

        for fabric_folder in fabric_folders:
            fabric_name = (
                fabric_folder.name[3:]
                if len(fabric_folder.name) > 3
                else fabric_folder.name
            )

            fabric_index = (self.fabric_to_index[fabric_name])

            for defect_folder in fabric_folder.iterdir():
                if not defect_folder.is_dir():
                    continue
                defect_full_name = (defect_folder.name)
                defect_index_number = int(defect_full_name.split(".")[0])
                defect_name = (defect_full_name.split(". ", 1)[1])

                expected_index = (self.defect_to_index[defect_name])

                if (defect_index_number != expected_index):
                    raise ValueError("Defect class index mismatch: " f"{defect_full_name}")

                for image_path in defect_folder.iterdir():
                    if (
                        image_path.is_file()
                        and image_path.suffix.lower()
                        in IMAGE_EXTENSIONS
                    ):

                        self.images.append((image_path, fabric_index, expected_index))

        if not self.images:
            raise ValueError(f"No images found in " f"{self.dataset_path}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_path, fabric_label, defect_label = (self.images[index])

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return (image, fabric_label, defect_label)