from pathlib import Path
import random
import shutil

SOURCE_DIR = Path("StitchingNet")
OUTPUT_DIR = Path("fabric_dataset")

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

RANDOM_SEED = 42

random.seed(RANDOM_SEED)

def get_images(folder):
    return [
        file
        for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]


def copy_images(images, destination):
    destination.mkdir(parents=True, exist_ok=True)

    for image in images:
        shutil.copy2(image, destination / image.name)

def main():

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {SOURCE_DIR}")

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output directory already exists: {OUTPUT_DIR}\n"
            "Delete it manually if you want to create the split again."
        )

    fabrics = [
        folder
        for folder in SOURCE_DIR.iterdir()
        if folder.is_dir()
    ]

    total_images = 0

    print("=" * 70)
    print("FABRIC DATASET SPLIT")
    print("=" * 70)

    for fabric_dir in sorted(fabrics):

        defect_dirs = [
            folder
            for folder in fabric_dir.iterdir()
            if folder.is_dir()
        ]

        for defect_dir in sorted(defect_dirs):

            images = get_images(defect_dir)

            if not images:
                continue

            random.shuffle(images)

            total = len(images)

            train_count = int(total * TRAIN_RATIO)
            validation_count = int(total * VALIDATION_RATIO)

            train_images = images[:train_count]

            validation_images = images[
                train_count:
                train_count + validation_count
            ]

            test_images = images[
                train_count + validation_count:
            ]

            relative_fabric = fabric_dir.name
            relative_defect = defect_dir.name

            train_destination = (
                OUTPUT_DIR
                / "train"
                / relative_fabric
                / relative_defect
            )

            validation_destination = (
                OUTPUT_DIR
                / "validation"
                / relative_fabric
                / relative_defect
            )

            test_destination = (
                OUTPUT_DIR
                / "test"
                / relative_fabric
                / relative_defect
            )

            copy_images(train_images, train_destination)

            copy_images(validation_images, validation_destination)

            copy_images(test_images, test_destination)

            total_images += total

            print(
                f"{fabric_dir.name:<28} "
                f"{defect_dir.name:<30} "
                f"{len(train_images):>3} / "
                f"{len(validation_images):>3} / "
                f"{len(test_images):>3}"
            )

    print("=" * 70)
    print(f"Total images: {total_images}")
    print("Train: 70%")
    print("Validation: 15%")
    print("Test: 15%")
    print("=" * 70)

    print()
    print(f"Dataset created at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()