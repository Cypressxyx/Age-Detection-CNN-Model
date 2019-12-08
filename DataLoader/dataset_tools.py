import os
import numpy as np
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img

" Load images into a numpy array"
def load_images(directory):
    target_size = (150, 150)
    filenames = os.listdir(directory)
    load_path = lambda path: load_img(os.path.join(directory, path), target_size=target_size) 
    images    = [img_to_array(load_path(image_path)) for image_path in filenames]
    return np.array(images)
