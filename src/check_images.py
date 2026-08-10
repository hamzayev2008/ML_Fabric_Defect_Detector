import os
from PIL import Image
for root, dirs, files in os.walk("dataset"):
    for image_name in files:
        image_path = os.path.join(root, image_name)
        try:
            with Image.open(image_path) as img:
                print(image_name, img.format)
        except (IOError, SyntaxError) as e:
            print(f"Invalid image: {image_path} - {e}")