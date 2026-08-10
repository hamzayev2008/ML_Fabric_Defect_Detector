import os
from PIL import Image

for root, dirs, files in os.walk("dataset"):
    for image_name in files:
        image_path = os.path.join(root, image_name)

        try:
            with Image.open(image_path) as img:
                if img.format != "JPEG":
                    converted_image = img.convert("RGB")
                else:
                    converted_image = None

            if converted_image is not None:
                converted_image.save(image_path, format="JPEG")
                print(f"Converted: {image_path}")

        except (IOError, SyntaxError) as e:
            print(f"Invalid image: {image_path} - {e}")