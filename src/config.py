TRAIN_DATASET_PATH = "fabric_dataset/train"
VALIDATION_DATASET_PATH = "fabric_dataset/validation"
TEST_DATASET_PATH = "fabric_dataset/test"
FABRIC_CLASSES = (
    "Cotton-Poly",
    "Linen-Poly",
    "Denim-Poly",
    "Velveteen-Poly",
    "Polyester-Poly",
    "Satin-Core",
    "Chiffon-Poly",
    "Nylon-Core",
    "Jacquard-Poly",
    "Oxford-Core",
    "Polyester (coated)-Core",
)
DEFECT_CLASSES = (
    "Normal",
    "Skipped stitch",
    "Broken stitch",
    "Pinched fabric",
    "Crooked seam",
    "Thread sagging",
    "Puckering",
    "Stain and damage",
    "Needle mark",
    "Bobbin thread pulling up",
    "Overlapped stitch",
)
MODEL_RESNET18_NAME = "resnet18"
MODEL_RESNET50_NAME = "resnet50"
MODEL_RESNET18_PATH = "classifier_resnet18.pth"
MODEL_RESNET50_PATH = "classifier_resnet50.pth"
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0001
EARLY_STOPPING = 5