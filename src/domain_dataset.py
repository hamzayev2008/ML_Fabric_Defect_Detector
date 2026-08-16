from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


class DomainDataset(Dataset):

    def __init__(self, dataset_path, transform=None):

        self.dataset_path = Path(dataset_path)

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}"
            )

        self.transform = transform

        self.class_names = [
            "Fabric",
            "Non-Fabric",
        ]

        self.class_to_index = {
            "fabric": 0,
            "non_fabric": 1,
        }

        self.images = []

        # ----------------------------------------------------
        # Find class directories
        # ----------------------------------------------------

        for class_name, class_index in self.class_to_index.items():

            class_dir = self.dataset_path / class_name

            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Class directory not found: {class_dir}"
                )

            for image_path in class_dir.rglob("*"):

                if (
                    image_path.is_file()
                    and image_path.suffix.lower()
                    in IMAGE_EXTENSIONS
                ):

                    self.images.append(
                        (
                            image_path,
                            class_index
                        )
                    )

        if not self.images:
            raise ValueError(
                f"No images found in {self.dataset_path}"
            )

        # ----------------------------------------------------
        # Sort for deterministic dataset order
        # ----------------------------------------------------

        self.images.sort(
            key=lambda item: str(item[0]).lower()
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_path, label = self.images[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label