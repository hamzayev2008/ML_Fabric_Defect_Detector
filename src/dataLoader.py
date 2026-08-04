from torch.utils.data import DataLoader
from config import IMAGE_SIZE, BATCH_SIZE, DATASET_PATH
from dataset import TeddyDataset

train_dataset = TeddyDataset(
    dataset_path = DATASET_PATH,
    image_size = IMAGE_SIZE,
    augmentation = None
)

train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    drop_last = False
)