from pathlib import Path
from PIL import Image
import imagehash


DATASET_PATH = Path("fabric_dataset")

SPLITS = ["train", "validation", "test"]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

HASH_DISTANCE = 5


def get_image_hash(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        return imagehash.phash(image)

    except Exception as error:
        print(f"Could not read: {image_path}")
        print(f"Error: {error}")
        return None


def collect_images(split):
    images = []

    split_path = DATASET_PATH / split

    for path in split_path.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            images.append(path)

    return images


def main():

    print("=" * 70)
    print("FINAL DATASET LEAKAGE CHECK")
    print("=" * 70)

    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found: {DATASET_PATH}")
        return

    split_images = {}

    for split in SPLITS:

        images = collect_images(split)

        split_images[split] = images

        print()
        print(f"{split.upper()}: {len(images)} images")

    print()
    print("=" * 70)
    print("Creating hashes...")
    print("=" * 70)

    split_hashes = {}

    for split in SPLITS:

        hashes = {}

        for image_path in split_images[split]:

            image_hash = get_image_hash(image_path)

            if image_hash is not None:
                hashes[image_path] = image_hash

        split_hashes[split] = hashes

        print(
            f"{split.upper()}: "
            f"{len(hashes)} hashes created"
        )

    print()
    print("=" * 70)
    print("CHECKING EXACT DUPLICATES")
    print("=" * 70)

    exact_leaks = 0

    checked_pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    for split_a, split_b in checked_pairs:

        hashes_a = split_hashes[split_a]
        hashes_b = split_hashes[split_b]

        hash_to_path_a = {
            str(image_hash): path
            for path, image_hash in hashes_a.items()
        }

        for path_b, hash_b in hashes_b.items():

            hash_string = str(hash_b)

            if hash_string in hash_to_path_a:

                exact_leaks += 1

                print()
                print("EXACT DUPLICATE FOUND!")

                print(f"{split_a}: {hash_to_path_a[hash_string]}")
                print(f"{split_b}: {path_b}")

    print()
    print("=" * 70)
    print("CHECKING SIMILAR IMAGES")
    print("=" * 70)

    similar_leaks = 0

    for split_a, split_b in checked_pairs:

        hashes_a = split_hashes[split_a]
        hashes_b = split_hashes[split_b]

        for path_a, hash_a in hashes_a.items():

            for path_b, hash_b in hashes_b.items():

                distance = hash_a - hash_b

                if distance <= HASH_DISTANCE:

                    similar_leaks += 1

                    print()
                    print("SIMILAR IMAGES FOUND!")

                    print(
                        f"{split_a}: {path_a}"
                    )

                    print(
                        f"{split_b}: {path_b}"
                    )

                    print(
                        f"Hash distance: {distance}"
                    )

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(f"Exact duplicates: {exact_leaks}")
    print(f"Similar image pairs: {similar_leaks}")

    print()

    if exact_leaks == 0 and similar_leaks == 0:

        print("SUCCESS!")
        print(
            "No cross-split image leakage was detected."
        )

    elif exact_leaks == 0:

        print(
            "No exact duplicates were found, "
            "but similar images were detected."
        )

    else:

        print(
            "WARNING!"
        )

        print(
            "Potential dataset leakage was detected."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()