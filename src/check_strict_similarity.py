from pathlib import Path
from PIL import Image
import imagehash
from collections import Counter

DATASET = Path("fabric_dataset")
HASH_DISTANCE = 2

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def get_images(split):
    split_path = DATASET / split

    return [
        path
        for path in split_path.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def get_hash(path):
    try:
        image = Image.open(path).convert("RGB")
        return imagehash.phash(image)

    except Exception:
        return None


def check_split_pair(images_a, name_a, images_b, name_b):
    print()
    print("=" * 70)
    print(f"{name_a.upper()} <-> {name_b.upper()}")
    print("=" * 70)

    hashes_a = {}
    hashes_b = {}

    for path in images_a:
        h = get_hash(path)
        if h is not None:
            hashes_a[path] = h

    for path in images_b:
        h = get_hash(path)
        if h is not None:
            hashes_b[path] = h

    distances = Counter()
    suspicious = []

    for path_a, hash_a in hashes_a.items():

        for path_b, hash_b in hashes_b.items():

            distance = hash_a - hash_b

            if distance <= HASH_DISTANCE:
                distances[distance] += 1

                suspicious.append(
                    (
                        distance,
                        path_a,
                        path_b
                    )
                )

    print(f"Images in {name_a}: {len(images_a)}")
    print(f"Images in {name_b}: {len(images_b)}")

    print()
    print("DISTANCE COUNTS")
    print("-" * 40)

    for distance in range(HASH_DISTANCE + 1):
        print(
            f"Distance {distance}: "
            f"{distances[distance]}"
        )

    print()
    print("SUSPICIOUS PAIRS")
    print("-" * 40)

    suspicious.sort(key=lambda x: x[0])

    for distance, path_a, path_b in suspicious:

        print()
        print(f"Distance: {distance}")
        print(f"{name_a}: {path_a}")
        print(f"{name_b}: {path_b}")

    return distances, suspicious


def main():

    print("=" * 70)
    print("STRICT CROSS-SPLIT SIMILARITY CHECK")
    print("=" * 70)

    train = get_images("train")
    validation = get_images("validation")
    test = get_images("test")

    total_distances = Counter()

    results = [
        check_split_pair(
            train,
            "train",
            validation,
            "validation"
        ),

        check_split_pair(
            train,
            "train",
            test,
            "test"
        ),

        check_split_pair(
            validation,
            "validation",
            test,
            "test"
        )
    ]

    for distances, _ in results:
        total_distances.update(distances)

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for distance in range(HASH_DISTANCE + 1):
        print(
            f"Distance {distance}: "
            f"{total_distances[distance]}"
        )

    print()

    if total_distances[0] > 0:

        print("WARNING!")
        print("Exact perceptual duplicates were found across splits.")

    elif total_distances[1] > 0:

        print("WARNING!")
        print("Extremely similar images were found across splits.")

    elif total_distances[2] > 0:

        print("CAUTION")
        print(
            "Very similar images were found, "
            "but they are not necessarily duplicates."
        )

    else:

        print("GOOD")
        print(
            "No highly similar images were found "
            "across different splits."
        )


if __name__ == "__main__":
    main()