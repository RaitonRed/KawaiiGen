import os
import numpy as np
from PIL import Image
def load_images(path, img_size=(64, 64)):
    images = []
    for file in os.listdir(path):
        if file.endswith('.jpg') or file.endswith('.png'):
            img = Image.open(os.path.join(path, file)).resize(img_size)
            img = np.array(img).astype('float32') / 255.0
            images.append(img)
    return np.array(images)