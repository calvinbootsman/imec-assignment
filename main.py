import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import json
import cv2
from dataset_loader import *


class NeuralNetwork(nn.Module):
    def __init__(self, grid_size=40, num_classes=1, numb_bounding_boxes=1):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.numb_bounding_boxes = numb_bounding_boxes
        
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = DatasetLoader(device, 'highway', num_bounding_boxes=10)
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)


model = NeuralNetwork().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

num_epochs = 10
for epoch in range(num_epochs):
    # model.train()
    for data_item, targets in dataset:
        image = data_item
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')