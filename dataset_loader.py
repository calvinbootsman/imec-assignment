import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os 
import random
import json
import cv2
from constants import *
import numpy as np

class DatasetLoader:
    def __init__(self, driving_style='highway', max_images=-1, num_boxes_per_cell=2, grid_size=7):
        self.data_path = f'train/{driving_style}'
        self.image_paths = []
        self.data_set = []
        self.max_images = max_images

        self.B = num_boxes_per_cell
        self.S = grid_size
        self.C = constants.NUM_OF_CLASSES

        with open('preprocessed_data/image_paths.txt', 'r') as file:
            self.image_paths = [line.strip() for line in file.readlines()]
        self.image_paths = self.image_paths[:max_images] if max_images > 0 else self.image_paths
        
    def __getitem__(self, index):
        path = f'preprocessed_data/sample_{index:06d}.pth'
        if not os.path.exists(path):
             raise IndexError("Index out of range")
    
        data_entry = torch.load(path)
        image_tensor = data_entry['image']
        target = data_entry['target']
        radar = data_entry['radar']
        camera = data_entry['camera']

        return image_tensor, target, radar, camera

    def __len__(self):
        return len(self.image_paths)

    def get_original_image(self, index):
        """
        Returns the original image at the specified index.
        """
        if index >= len(self.image_paths):
            raise IndexError("Index out of range")
        image_path = self.image_paths[index]
        original_image = cv2.imread(image_path)
        return original_image
