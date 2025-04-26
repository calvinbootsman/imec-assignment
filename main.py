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
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.input_dims) 
            cnn_out_shape = self.cnn(dummy_input).shape
            self.flattened_size = cnn_out_shape[1] * cnn_out_shape[2] * cnn_out_shape[3]

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(self.flattened_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.1), 
            nn.Linear(1024, self.numb_bounding_boxes * self.output_features_per_box)
        )

    def forward(self, x):
        features = self.cnn(x)
        predictions = self.fc(features)
        predictions = predictions.view(-1, self.numb_bounding_boxes, self.output_features_per_box)
        return predictions
    

labels = ['SIZE_VEHICLE_M', 'SIZE_VEHICLE_XL', 'TRAFFIC_SIGN', 'TRAFFIC_LIGHT', 'PEDESTRIAN', 'TWO_WHEEL_WITHOUT_RIDER', 'RIDER']

def draw_bounding_box(image, tensors):
    height, width, _ = image.shape

    for box in tensors:
        x_center = box[1] * width
        y_center = box[2] * height
        width_box = box[3] * width
        height_box = box[4] * height
        x1 = int(x_center - width_box / 2)
        y1 = int(y_center - height_box / 2)
        x2 = int(x_center + width_box / 2)
        y2 = int(y_center + height_box / 2)

        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    return image
if __name__ == "__main__":
    torch.manual_seed(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = DatasetLoader('highway', num_bounding_boxes=constants.MAX_NUM_BBOXES)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])

    num_workers = 4
    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=num_workers, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=num_workers, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=num_workers, batch_size=32, shuffle=False)


    model = NeuralNetwork(grid_size=30, num_classes=6, numb_bounding_boxes=constants.MAX_NUM_BBOXES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    max_no_improvement = 5
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=max_no_improvement)

    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (batch_data_item, batch_targets) in enumerate(train_loader):
            image = batch_data_item.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            outputs = model(image)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99: # Print stats every 100 batches
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Batch Loss: {(running_loss/100):.4f}')
                running_loss = 0.0
            if i % 25 == 0:
                scheduler.step(loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        # Validation
        model.eval()
        random_index = random.randint(0, len(val_dataset) - 1)
        random_sample = val_dataset[random_index]
        _, target = random_sample
        image = dataset.get_original_image(random_index)
        output_image = draw_bounding_box(image, target)
        cv2.imshow('Output', output_image)
        cv2.waitKey(0)




        # val_loss = 0.0
        # with torch.no_grad():
        #     for i, (batch_data_item, batch_targets) in enumerate(val_loader):
        #         image = batch_data_item.to(device)
        #         batch_targets = batch_targets.to(device)
        #         outputs = model(image)
        #         loss = criterion(outputs, batch_targets)
        #         val_loss += loss.item()
        # val_loss /= len(val_loader)
        

    