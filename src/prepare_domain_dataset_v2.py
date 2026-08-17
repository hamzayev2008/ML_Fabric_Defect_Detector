from pathlib import Path
import shutil
import random


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FABRIC_SOURCE = PROJECT_ROOT / "domain_dataset" / "fabric_composition"
NON_FABRIC_SOURCE = PROJECT_ROOT / "domain_dataset" / "non_fabric"

OUTPUT = PROJECT_ROOT / "domain_dataset" / "gate_v2"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SEED = 42
random.seed(SEED)


def get_images(directory):
    return [
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def split_images(images):
    images = list(images)
    random.shuffle(images)

    n = len(images)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    return (
        images[:train_end],
        images[train_end:val_end],
        images[val_end:]
    )


def copy_images(images, destination):
    destination.mkdir(parents=True, exist_ok=True)

    for i, image in enumerate(images):
        new_name = f"{i:06d}_{image.name}"
        shutil.copy2(image, destination / new_name)


def main():

    print("=" * 70)
    print("PREPARING DOMAIN DATASET V2")
    print("=" * 70)

    # --------------------------------------------------------
    # Check sources
    # --------------------------------------------------------

    if not FABRIC_SOURCE.exists():
        raise FileNotFoundError(
            f"Fabric source not found: {FABRIC_SOURCE}"
        )

    if not NON_FABRIC_SOURCE.exists():
        raise FileNotFoundError(
            f"Non-Fabric source not found: {NON_FABRIC_SOURCE}"
        )

    # --------------------------------------------------------
    # Collect images
    # --------------------------------------------------------

    print("\nCollecting Fabric images...")
    fabric_images = get_images(FABRIC_SOURCE)

    print(f"FabricsComposition: {len(fabric_images)}")

    print("\nCollecting Non-Fabric images...")
    non_fabric_images = get_images(NON_FABRIC_SOURCE)

    print(f"Caltech Non-Fabric:  {len(non_fabric_images)}")

    if not fabric_images:
        raise RuntimeError("No Fabric images found.")

    if not non_fabric_images:
        raise RuntimeError("No Non-Fabric images found.")

    # --------------------------------------------------------
    # Do NOT balance to 50/50
    # Use all available images.
    # --------------------------------------------------------

    print("\nDataset sizes:")
    print(f"Fabric:              {len(fabric_images)}")
    print(f"Non-Fabric:          {len(non_fabric_images)}")

    # --------------------------------------------------------
    # Split independently
    # --------------------------------------------------------

    fabric_train, fabric_val, fabric_test = split_images(
        fabric_images
    )

    non_train, non_val, non_test = split_images(
        non_fabric_images
    )

    # --------------------------------------------------------
    # Remove old output if present
    # --------------------------------------------------------

    if OUTPUT.exists():
        print(f"\nRemoving existing output: {OUTPUT}")
        shutil.rmtree(OUTPUT)

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    print("\nPreparing output directories...")

    for split in ["train", "validation", "test"]:
        (OUTPUT / split / "fabric").mkdir(
            parents=True,
            exist_ok=True
        )

        (OUTPUT / split / "non_fabric").mkdir(
            parents=True,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Copy Fabric
    # --------------------------------------------------------

    print("\nCopying Fabric images...")

    copy_images(
        fabric_train,
        OUTPUT / "train" / "fabric"
    )

    copy_images(
        fabric_val,
        OUTPUT / "validation" / "fabric"
    )

    copy_images(
        fabric_test,
        OUTPUT / "test" / "fabric"
    )

    # --------------------------------------------------------
    # Copy Non-Fabric
    # --------------------------------------------------------

    print("Copying Non-Fabric images...")

    copy_images(
        non_train,
        OUTPUT / "train" / "non_fabric"
    )

    copy_images(
        non_val,
        OUTPUT / "validation" / "non_fabric"
    )

    copy_images(
        non_test,
        OUTPUT / "test" / "non_fabric"
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DOMAIN DATASET V2 CREATED")
    print("=" * 70)

    print(
        f"TRAIN       : fabric={len(fabric_train):5d}, "
        f"non_fabric={len(non_train):5d}, "
        f"total={len(fabric_train) + len(non_train):5d}"
    )

    print(
        f"VALIDATION  : fabric={len(fabric_val):5d}, "
        f"non_fabric={len(non_val):5d}, "
        f"total={len(fabric_val) + len(non_val):5d}"
    )

    print(
        f"TEST        : fabric={len(fabric_test):5d}, "
        f"non_fabric={len(non_test):5d}, "
        f"total={len(fabric_test) + len(non_test):5d}"
    )

    total = (
        len(fabric_images) +
        len(non_fabric_images)
    )

    print("-" * 70)
    print(f"Total images: {total}")
    print(f"Output:       {OUTPUT}")
    print("-" * 70)
    print("Only FabricsComposition + Caltech Non-Fabric were used.")
    print("Original datasets were not modified.")


if __name__ == "__main__":
    main()