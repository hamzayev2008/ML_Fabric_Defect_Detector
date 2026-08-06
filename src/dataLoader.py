from torch.utils.data import DataLoader
from config import IMAGE_SIZE, BATCH_SIZE, TRAIN_DATASET_PATH, VALIDATION_DATASET_PATH
from dataset import TeddyDataset

train_dataset = TeddyDataset(
    dataset_path = TRAIN_DATASET_PATH,
    image_size = IMAGE_SIZE,
    augmentation = None
)

train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    drop_last = False
)

validation_dataset = TeddyDataset(
    dataset_path = VALIDATION_DATASET_PATH,
    image_size = IMAGE_SIZE,
    augmentation = None
)

validation_loader = DataLoader(
    dataset = validation_dataset,
    batch_size = BATCH_SIZE,
    shuffle = False
)