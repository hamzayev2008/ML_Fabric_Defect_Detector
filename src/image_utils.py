import cv2
import numpy as np

def load_image_from_bytes(image_bytes, image_size, transform):
    numpy_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
    return process_image(image, image_size, transform)

def load_image(path, image_size, transform):
    image = cv2.imread(path)
    return process_image(image, image_size, transform)

def process_image(image, image_size, transform):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size))
    image = transform(image)
    return image