import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import json
import cv2
from dataset_loader import *


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__(grid_size=40, num_classes=1, numb_bounding_boxes=1)
        self.grid_size = 40
        self.num_classes = 1
        self.numb_bounding_boxes = 1
        
        self.cnn = nn.Sequential(
            # Based on the VGG16 architecture
            # https://arxiv.org/pdf/1409.1556.pdf
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((self.grid_size,self.grid_size)),
            # self.numb_bounding_boxes * 5 because of the 4 coordinates of the bounding box and 1 for the confidence score
            nn.Conv2d(256, self.num_classes + (self.numb_bounding_boxes * 5), kernel_size=1, stride=1, padding=1),
        )

    def forward(self, x):
        predictions = self.cnn(x)
        predictions = predictions.permute(0, 2, 3, 1)  # Change the order of dimensions to (batch_size, grid_size, grid_size, num_classes + (numb_bounding_boxes * 5))
        return predictions
    

labels = ['SIZE_VEHICLE_M', 'SIZE_VEHICLE_XL', 'TRAFFIC_SIGN', 'TRAFFIC_LIGHT', 'PEDESTRIAN', 'TWO_WHEEL_WITHOUT_RIDER', 'RIDER']

def draw_bounding_box(image, boxes):
    for box in boxes:
        cv2.rectangle(image, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), (0, 255, 0), 2)
    return image

dataset = DatasetLoader('highway')
items = dataset_loader.get_random_pictures(1)
data_entry = items[0]
image = draw_bounding_box(data_entry.image, data_entry.bounding_box)
cv2.imshow('image', image)
cv2.waitKey(0)