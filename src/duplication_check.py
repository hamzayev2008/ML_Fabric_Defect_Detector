from pathlib import Path
from PIL import Image
import imagehash


DATASET_PATH = Path("fabric_dataset")

SPLITS = [
    "train",
    "validation",
    "test"
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

# Чем меньше значение, тем сильнее похожи изображения.
# 0 = практически одинаковые.
HASH_THRESHOLD = 5


def collect_images(split):
    split_path = DATASET_PATH / split

    images = []

    for path in split_path.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            images.append(path)

    return images


def calculate_hash(path):
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            return imagehash.phash(image)

    except Exception as error:
        print(f"Could not read: {path}")
        print(error)

        return None


print("=" * 70)
print("PERCEPTUAL DUPLICATE CHECK")
print("=" * 70)


# ---------------------------------------------------------
# Collect images
# ---------------------------------------------------------

images_by_split = {}

for split in SPLITS:

    images = collect_images(split)

    images_by_split[split] = images

    print(
        f"{split.capitalize():12}: "
        f"{len(images)} images"
    )


print("=" * 70)


# ---------------------------------------------------------
# Calculate hashes
# ---------------------------------------------------------

hashes_by_split = {}

for split in SPLITS:

    print(f"Calculating perceptual hashes: {split}...")

    hashes = []

    for index, image_path in enumerate(
        images_by_split[split],
        start=1
    ):

        image_hash = calculate_hash(image_path)

        if image_hash is not None:
            hashes.append(
                (image_hash, image_path)
            )

        if index % 500 == 0:
            print(
                f"  {index}/{len(images_by_split[split])}"
            )

    hashes_by_split[split] = hashes


print("=" * 70)


# ---------------------------------------------------------
# Compare splits
# ---------------------------------------------------------

total_suspicious = 0


for i in range(len(SPLITS)):

    for j in range(i + 1, len(SPLITS)):

        split_a = SPLITS[i]
        split_b = SPLITS[j]

        print(
            f"\nChecking: "
            f"{split_a} <-> {split_b}"
        )

        suspicious_pairs = []

        for hash_a, path_a in hashes_by_split[split_a]:

            for hash_b, path_b in hashes_by_split[split_b]:

                distance = hash_a - hash_b

                if distance <= HASH_THRESHOLD:

                    suspicious_pairs.append(
                        (
                            distance,
                            path_a,
                            path_b
                        )
                    )

        suspicious_pairs.sort(
            key=lambda x: x[0]
        )

        print(
            f"Suspicious pairs: "
            f"{len(suspicious_pairs)}"
        )

        for distance, path_a, path_b in suspicious_pairs[:20]:

            print()
            print(f"Distance: {distance}")
            print(f"  {path_a}")
            print(f"  {path_b}")

        total_suspicious += len(
            suspicious_pairs
        )


print("\n" + "=" * 70)


if total_suspicious == 0:

    print(
        "GOOD: No suspiciously similar images "
        "were found between dataset splits."
    )

else:

    print(
        f"WARNING: Found "
        f"{total_suspicious} suspicious pairs."
    )

    print(
        "Inspect these images before trusting "
        "the validation/test results."
    )


print("=" * 70)