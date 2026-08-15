import shutil
import random
from pathlib import Path
from PIL import Image
import imagehash

SOURCE_DATASET = Path(r"C:\Users\user\Downloads\StitchingNet\StitchingNet")
OUTPUT_DATASET = Path("fabric_dataset")
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp",}
HASH_DISTANCE = 5

random.seed(SEED)

def get_image_hash(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        return imagehash.phash(image)

    except Exception:
        return None

def create_similarity_groups(image_paths):
    hashes = {}

    for image_path in image_paths:
        image_hash = get_image_hash(image_path)

        if image_hash is not None:
            hashes[image_path] = image_hash

    groups = []

    for image_path in image_paths:

        if image_path not in hashes:
            groups.append([image_path])
            continue

        # Find an existing group whose representative is similar.
        added_to_group = False

        for group in groups:

            representative = group[0]

            if representative not in hashes:
                continue

            distance = hashes[image_path] - hashes[representative]

            if distance <= HASH_DISTANCE:
                group.append(image_path)
                added_to_group = True
                break

        if not added_to_group:
            groups.append([image_path])

    return groups


def split_groups(groups):

    random.shuffle(groups)

    total_images = sum(
        len(group)
        for group in groups
    )

    train_target = total_images * TRAIN_RATIO
    validation_target = total_images * VALIDATION_RATIO

    train = []
    validation = []
    test = []

    train_count = 0
    validation_count = 0

    for group in groups:

        group_size = len(group)

        # Put the entire group into one split.
        if train_count + group_size <= train_target:

            train.extend(group)
            train_count += group_size

        elif validation_count + group_size <= validation_target:

            validation.extend(group)
            validation_count += group_size

        else:

            test.extend(group)

    return train, validation, test


def copy_images(image_paths, split_name, fabric_name, defect_name):

    destination = (
        OUTPUT_DATASET
        / split_name
        / fabric_name
        / defect_name
    )

    destination.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:

        destination_path = (
            destination
            / image_path.name
        )

        counter = 1

        while destination_path.exists():

            destination_path = (
                destination
                / f"{image_path.stem}_{counter}"
                f"{image_path.suffix}"
            )

            counter += 1

        shutil.copy2(image_path, destination_path)

def main():
    print("=" * 70)
    print("LEAKAGE-SAFE FABRIC DATASET SPLIT")
    print("=" * 70)

    if not SOURCE_DATASET.exists():
        print(f"ERROR: Dataset not found: " f"{SOURCE_DATASET}")
        return

    # Remove the previous split.
    if OUTPUT_DATASET.exists():

        print(f"Removing existing dataset: " f"{OUTPUT_DATASET}")

        shutil.rmtree(OUTPUT_DATASET)

    total_images = 0
    total_groups = 0

    for fabric_dir in sorted(SOURCE_DATASET.iterdir()):

        if not fabric_dir.is_dir():
            continue

        fabric_name = fabric_dir.name

        for defect_dir in sorted(fabric_dir.iterdir()):

            if not defect_dir.is_dir():
                continue

            defect_name = defect_dir.name

            image_paths = [
                path
                for path in defect_dir.iterdir()
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in IMAGE_EXTENSIONS
                )
            ]

            if not image_paths:
                continue

            print()
            print(f"{fabric_name} / " f"{defect_name}")
            print(f"Images: {len(image_paths)}")
            print("Creating similarity groups...")
            groups = create_similarity_groups(image_paths)
            print(f"Similarity groups: " f"{len(groups)}")
            train, validation, test = (split_groups(groups))
            print(
                f"Train: {len(train):4d} | "
                f"Validation: {len(validation):4d} | "
                f"Test: {len(test):4d}"
            )
            copy_images(train, "train", fabric_name, defect_name)
            copy_images(validation, "validation", fabric_name, defect_name)
            copy_images(test, "test", fabric_name, defect_name)
            total_images += len(image_paths)
            total_groups += len(groups)
    print()
    print("=" * 70)
    print("SPLIT FINISHED")
    print("=" * 70)
    print(f"Total images: {total_images}")
    print(f"Total similarity groups: " f"{total_groups}")
    print(f"Dataset created at: " f"{OUTPUT_DATASET}")
    print("=" * 70)

if __name__ == "__main__":
    main()