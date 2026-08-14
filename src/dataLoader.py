from torch.utils.data import DataLoader

from config import (
    TRAIN_DATASET_PATH,
    VALIDATION_DATASET_PATH,
    TEST_DATASET_PATH,
    BATCH_SIZE
)

from dataset import FabricDataset

train_dataset = FabricDataset(
    dataset_path=TRAIN_DATASET_PATH,
    augmentation=True
)

validation_dataset = FabricDataset(
    dataset_path=VALIDATION_DATASET_PATH,
    augmentation=False
)

test_dataset = FabricDataset(
    dataset_path=TEST_DATASET_PATH,
    augmentation=False
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False
)

validation_loader = DataLoader(
    dataset=validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)