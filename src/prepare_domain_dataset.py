from pathlib import Path
import random
import shutil


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FABRIC_DATASET = PROJECT_ROOT / "fabric_dataset"

NON_FABRIC_DATASET = (
    PROJECT_ROOT
    / "domain_dataset"
    / "non_fabric"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "domain_dataset"
    / "gate"
)

SPLITS = (
    "train",
    "validation",
    "test",
)

NON_FABRIC_SPLIT_COUNTS = {
    "train": 10_500,
    "validation": 2_304,
    "test": 2_300,
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

SEED = 42


def get_images(directory):
    return sorted(
        [
            path
            for path in directory.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        ]
    )


def make_unique_name(path, root):
    relative = path.relative_to(root)

    parts = list(relative.parts)

    # Remove the file extension from the final part.
    filename = Path(parts[-1])

    stem = filename.stem
    suffix = filename.suffix

    # Include the relative directory information so that
    # files with equal names cannot overwrite each other.
    prefix = "_".join(
        part.replace(" ", "_")
        for part in parts[:-1]
    )

    if prefix:
        return f"{prefix}_{stem}{suffix}"

    return f"{stem}{suffix}"


def copy_images(images, source_root, destination):
    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    for image_path in images:

        filename = make_unique_name(
            image_path,
            source_root
        )

        shutil.copy2(
            image_path,
            destination / filename
        )


def main():

    print("=" * 70)
    print("PREPARING DOMAIN DATASET")
    print("=" * 70)

    # ========================================================
    # CHECK SOURCES
    # ========================================================

    if not FABRIC_DATASET.exists():
        raise FileNotFoundError(
            f"Fabric dataset not found: {FABRIC_DATASET}"
        )

    if not NON_FABRIC_DATASET.exists():
        raise FileNotFoundError(
            f"Non-fabric dataset not found: "
            f"{NON_FABRIC_DATASET}"
        )

    # ========================================================
    # COLLECT FABRIC IMAGES
    # ========================================================

    print()
    print("Collecting fabric images...")

    fabric_images = {}

    for split in SPLITS:

        split_dir = FABRIC_DATASET / split

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Fabric split not found: {split_dir}"
            )

        images = get_images(split_dir)

        fabric_images[split] = images

        print(
            f"{split:12s}: {len(images):5d}"
        )

    # ========================================================
    # COLLECT NON-FABRIC IMAGES
    # ========================================================

    print()
    print("Collecting non-fabric images...")

    non_fabric_images = get_images(
        NON_FABRIC_DATASET
    )

    print(
        f"Available non-fabric: "
        f"{len(non_fabric_images)}"
    )

    requested_non_fabric = sum(
        NON_FABRIC_SPLIT_COUNTS.values()
    )

    if len(non_fabric_images) < requested_non_fabric:

        raise ValueError(
            f"Need {requested_non_fabric} non-fabric images, "
            f"but only {len(non_fabric_images)} are available."
        )

    # ========================================================
    # SHUFFLE NON-FABRIC
    # ========================================================

    random.seed(SEED)

    random.shuffle(
        non_fabric_images
    )

    # ========================================================
    # SPLIT NON-FABRIC
    # ========================================================

    non_fabric_splits = {}

    start = 0

    for split in SPLITS:

        count = NON_FABRIC_SPLIT_COUNTS[split]

        end = start + count

        non_fabric_splits[split] = (
            non_fabric_images[start:end]
        )

        start = end

    # ========================================================
    # PREPARE OUTPUT
    # ========================================================

    print()
    print("Preparing output directories...")

    if OUTPUT_ROOT.exists():

        print(
            f"Removing previous generated dataset: "
            f"{OUTPUT_ROOT}"
        )

        shutil.rmtree(
            OUTPUT_ROOT
        )

    # ========================================================
    # COPY DATA
    # ========================================================

    print()
    print("Copying images...")

    for split in SPLITS:

        fabric_output = (
            OUTPUT_ROOT
            / split
            / "fabric"
        )

        non_fabric_output = (
            OUTPUT_ROOT
            / split
            / "non_fabric"
        )

        copy_images(
            fabric_images[split],
            FABRIC_DATASET / split,
            fabric_output
        )

        copy_images(
            non_fabric_splits[split],
            NON_FABRIC_DATASET,
            non_fabric_output
        )

        print()
        print(f"{split.upper()}")

        print(
            f"  fabric:     "
            f"{len(fabric_images[split])}"
        )

        print(
            f"  non_fabric: "
            f"{len(non_fabric_splits[split])}"
        )

    # ========================================================
    # FINAL CHECK
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL DOMAIN DATASET")
    print("=" * 70)

    total = 0

    for split in SPLITS:

        fabric_count = len(
            get_images(
                OUTPUT_ROOT
                / split
                / "fabric"
            )
        )

        non_fabric_count = len(
            get_images(
                OUTPUT_ROOT
                / split
                / "non_fabric"
            )
        )

        split_total = (
            fabric_count
            + non_fabric_count
        )

        total += split_total

        print(
            f"{split:12s}: "
            f"fabric={fabric_count:5d}, "
            f"non_fabric={non_fabric_count:5d}, "
            f"total={split_total:5d}"
        )

    print("-" * 70)
    print(
        f"Total images: {total}"
    )

    print()
    print(
        f"Output: {OUTPUT_ROOT}"
    )

    print()
    print(
        "Original fabric_dataset was not modified."
    )

    print(
        "Original non_fabric dataset was not modified."
    )


if __name__ == "__main__":
    main()