import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import json
import cv2
from dataset_loader import *
from constants import *


class NeuralNetwork(nn.Module):
    def __init__(self, grid_size=40, num_classes=1, numb_bounding_boxes=1):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.numb_bounding_boxes = numb_bounding_boxes
        self.output_features_per_box = 6 # 4 coordinates + 1 confidence score + 1 class score
        self.input_dims = (3, constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH)

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
        #     nn.AdaptiveMaxPool2d((self.grid_size,self.grid_size)),
        #     # self.numb_bounding_boxes * 5 because of the 4 coordinates of the bounding box and 1 for the confidence score
        #     nn.Conv2d(256, self.num_classes + (self.numb_bounding_boxes * 5), kernel_size=1, stride=1, padding=1),
        )

         # Calculate the size after the CNN layers
        # We need a dummy tensor pass to determine the flattened size dynamically
        # Or calculate it manually based on input_dims and CNN architecture
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.input_dims) 
            cnn_out_shape = self.cnn(dummy_input).shape
            self.flattened_size = cnn_out_shape[1] * cnn_out_shape[2] * cnn_out_shape[3]

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(), # Flatten the output of CNN: (B, 256, H/16, W/16) -> (B, 256*H/16*W/16)
            nn.Linear(self.flattened_size, 1024), # Example intermediate layer
            nn.ReLU(),
            nn.Dropout(0.5), # Optional dropout for regularization
            # Output layer: Predict parameters for N boxes
            # Total outputs = num_boxes * features_per_box
            nn.Linear(1024, self.numb_bounding_boxes * self.output_features_per_box)
        )

    def forward(self, x):
        features = self.cnn(x)
        predictions = self.fc(features)
        predictions = predictions.view(-1, self.numb_bounding_boxes, self.output_features_per_box)
        return predictions
    

labels = ['SIZE_VEHICLE_M', 'SIZE_VEHICLE_XL', 'TRAFFIC_SIGN', 'TRAFFIC_LIGHT', 'PEDESTRIAN', 'TWO_WHEEL_WITHOUT_RIDER', 'RIDER']

def draw_bounding_box(image, boxes):
    for box in boxes:
        cv2.rectangle(image, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), (0, 255, 0), 2)
    return image
torch.manual_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = DatasetLoader(device, 'highway', num_bounding_boxes=constants.MAX_NUM_BBOXES)
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)


model = NeuralNetwork(grid_size=30, num_classes=6, numb_bounding_boxes=constants.MAX_NUM_BBOXES).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch_data_item, batch_targets in train_loader:
        image = batch_data_item
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()
        print(f'Batch Loss: {loss.item():.4f}')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')