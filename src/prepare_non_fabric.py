from pathlib import Path
import random
import shutil
import math

SOURCE_DIR = Path(
    r"C:\Users\user\Downloads\256_ObjectCategories\256_ObjectCategories"
)

OUTPUT_DIR = Path("domain_dataset/non_fabric")

TARGET_IMAGES = 15_000
SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

random.seed(SEED)


def get_images(category_dir):
    return [
        path
        for path in category_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    ]


def main():

    print("=" * 70)
    print("PREPARING NON-FABRIC DATASET")
    print("=" * 70)

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Caltech dataset not found: {SOURCE_DIR}"
        )

    categories = sorted(
        [
            path
            for path in SOURCE_DIR.iterdir()
            if path.is_dir()
        ],
        key=lambda path: path.name
    )

    print(f"Available categories: {len(categories)}")

    category_images = {}

    for category in categories:

        images = get_images(category)

        if images:
            category_images[category.name] = images

    total_available = sum(
        len(images)
        for images in category_images.values()
    )

    print(f"Available images: {total_available}")
    print(f"Target images:    {TARGET_IMAGES}")

    if total_available < TARGET_IMAGES:
        raise ValueError(
            "Not enough images available."
        )

    # --------------------------------------------------------
    # Determine how many images to take from each category.
    # We start with an equal quota and redistribute unused
    # quota from small categories.
    # --------------------------------------------------------

    category_names = list(category_images.keys())

    selected = {
        category: []
        for category in category_names
    }

    remaining_categories = set(category_names)
    remaining_target = TARGET_IMAGES

    while remaining_categories:

        quota = math.ceil(
            remaining_target / len(remaining_categories)
        )

        categories_finished = []

        for category in remaining_categories:

            available = len(category_images[category])

            amount = min(
                quota,
                available
            )

            selected[category] = random.sample(
                category_images[category],
                amount
            )

            remaining_target -= amount

            if available <= quota:
                categories_finished.append(category)

        for category in categories_finished:
            remaining_categories.remove(category)

        # ----------------------------------------------------
        # If quota was too small for any category to be
        # finished, exit the loop.
        # ----------------------------------------------------

        if not categories_finished:
            break

    # --------------------------------------------------------
    # If the quota algorithm leaves a small remainder, fill it
    # from the remaining unused images.
    # --------------------------------------------------------

    selected_paths = [
        path
        for paths in selected.values()
        for path in paths
    ]

    selected_set = set(selected_paths)

    remaining_images = [
        image
        for images in category_images.values()
        for image in images
        if image not in selected_set
    ]

    random.shuffle(remaining_images)

    missing = TARGET_IMAGES - len(selected_paths)

    if missing > 0:

        selected_paths.extend(
            remaining_images[:missing]
        )

    if len(selected_paths) != TARGET_IMAGES:
        raise RuntimeError(
            f"Expected {TARGET_IMAGES} images, "
            f"but selected {len(selected_paths)}."
        )

    # --------------------------------------------------------
    # Recreate output directory.
    # This only modifies our generated non_fabric folder,
    # never the original Caltech dataset.
    # --------------------------------------------------------

    if OUTPUT_DIR.exists():
        print()
        print(f"Removing existing: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Copy images.
    # Keep source category in the filename to avoid collisions.
    # --------------------------------------------------------

    for index, image_path in enumerate(
        selected_paths,
        start=1
    ):

        category_name = image_path.parent.name

        destination = (
            OUTPUT_DIR
            / f"{index:05d}_{category_name}_{image_path.name}"
        )

        shutil.copy2(
            image_path,
            destination
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("NON-FABRIC DATASET CREATED")
    print("=" * 70)

    print(f"Images copied: {len(selected_paths)}")
    print(f"Output:        {OUTPUT_DIR}")

    print()
    print("Top category counts:")

    counts = {
        category: len(paths)
        for category, paths in selected.items()
        if paths
    }

    for category, count in sorted(
        counts.items(),
        key=lambda item: item[1],
        reverse=True
    )[:20]:

        print(
            f"{category:30s} {count:4d}"
        )

    print()
    print("Original Caltech dataset was not modified.")


if __name__ == "__main__":
    main()