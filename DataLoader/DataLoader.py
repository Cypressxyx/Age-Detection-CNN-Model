"""
A class to manage loading datasets
"""
import os
from DataLoader.dataset_tools import load_images

class Dataset:
    def __init__(self, base_dir):
        self.directory = base_dir 
        self.filenames = os.list_dir(base_dir)
        self.image_data = load_images(directory)
        self.num_images = len(self.image_data)

    "Load a simple 80 20 train test data split"
    def get_train_test_data(self):
        num_train   = int(self.num_images * .8)
        randomize_indices = np.random.permutation(self.num_images)

        train_data = self.image_data[p[:num_train]]
        #train_labels = self.image_data[p[:num_train]]
        train_labels = np.array([1])

        test_data  = self.image_data[p[num_train:]]
        test_labels = np.array([1])
        #train_labels

        return (train_data, train_labels), (test_data, test_labels)
