import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import *

class CarRecorgnitionNetwork(nn.Module):
    def __init__(self, num_classes=1, num_boxes=1):
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.output_features_per_cell = (num_classes* 5) + num_classes# 4 coordinates + 1 confidence score
        self.output_features_per_box = (num_boxes * 5) + num_classes
        self.input_dims = (3, constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH)
        self.output_dim = int(constants.GRID_SIZE * constants.GRID_SIZE * self.output_features_per_cell)

        CNN_OUTPUT_CHANNELS = 256
        self.cnn_output_dim = CNN_OUTPUT_CHANNELS * (constants.IMAGE_HEIGHT // 16) * (constants.IMAGE_WIDTH // 16)
        self.cnn = nn.Sequential(
            # Based on the VGG16 architecture
            # https://arxiv.org/pdf/1409.1556.pdf
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(CNN_OUTPUT_CHANNELS),
            nn.LeakyReLU(0.1),
        )

        flatten = nn.Flatten()

        self.dense_layers = nn.Sequential(
            nn.Linear(CNN_OUTPUT_CHANNELS * (constants.IMAGE_HEIGHT // 16) * (constants.IMAGE_WIDTH // 16), CNN_OUTPUT_CHANNELS),
            nn.BatchNorm1d(CNN_OUTPUT_CHANNELS),
            nn.LeakyReLU(0.1),
            
            nn.Linear(CNN_OUTPUT_CHANNELS, self.output_dim),
            nn.Sigmoid()
        )

        self.model = nn.Sequential(
            self.cnn,
            flatten,
            self.dense_layers,
            nn.Unflatten(1, (constants.GRID_SIZE, constants.GRID_SIZE, self.output_features_per_cell))
        )

    def forward(self, x):
        return self.model(x)
    
def difference(x, y):
    return torch.sum((y - x)**2)

def yolo_loss(target, predictions):
    """
    Custom YOLO loss function.
    Args:
        y_true (torch.Tensor): Ground truth tensor of shape (batch_size, S, S, B*5 + C).
        y_predictions (torch.Tensor): Predicted tensor of shape (batch_size, S, S, B*5 + C).
    Returns:
        torch.Tensor: Computed loss value.
    """
    pred_boxes = []
    pred_confidence = []

    B = constants.MAX_NUM_BBOXES
    S = constants.GRID_SIZE
    C = constants.NUM_OF_CLASSES
    mse = nn.MSELoss(reduction="sum")
    lambda_noobj = 0.5
    lambda_coord = 5
    for b in range(B):
        offset = b * 5
        pred_boxes.append(predictions[..., offset : offset+4])
        pred_confidence.append(predictions[..., offset+4 : offset+5])
    pred_classes = predictions[..., B*5 :]
    pred_distance = predictions[...,B*5 + 1:]

    target_box = target[..., 0:4] # [x, y, w, h]
    target_conf = target[..., 4:5] # Confidence (objectness) - should be 1 if object present, 0 otherwise
    target_classes = target[..., B*5 :] # One-hot encoded classes
    target_distance = target[...,B*5 + 1 :]

    exists_box = target_conf  # target_conf used as a mask

    # Loss Box coordinates (x, y, w, h)
    box_predictions = exists_box * pred_boxes[0]
    # Target box coordinates
    box_targets = exists_box * target_box
    box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
        torch.abs(box_predictions[..., 2:4] + 1e-6)
    )
    box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4] + 1e-6) # Target w,h are always >= 0

    # Calculate MSE loss for coordinates (x, y, sqrt(w), sqrt(h))
    # (BATCH, S, S, 4) -> scalar
    box_loss = mse(
        torch.flatten(box_predictions, end_dim=-2), # Flatten (BATCH*S*S, 4)
        torch.flatten(box_targets, end_dim=-2),
    )
    
    # Loss Object
    pred_conf_obj = exists_box * pred_confidence[0] # (BATCH, S, S, 1)
    target_conf_obj = exists_box * target_conf # Target is already 1 where object exists
    object_loss = mse(
            torch.flatten(pred_conf_obj), # Flatten (BATCH*S*S)
            torch.flatten(target_conf_obj)
    )

    # Loss No Object
    no_object_loss = 0.0
    no_exists_box = (1.0 - exists_box) # Mask for cells without objects
    for b in range(B):
            pred_conf_noobj = no_exists_box * pred_confidence[b]
            target_conf_noobj = no_exists_box * torch.zeros_like(target_conf) # Target is 0
            no_object_loss += mse(
                torch.flatten(pred_conf_noobj), # Flatten (BATCH*S*S)
                torch.flatten(target_conf_noobj) # Flatten (BATCH*S*S) - this is just zeros
            )
    # Loss Class
    pred_classes_obj = exists_box * pred_classes # (BATCH, S, S, C)
    target_classes_obj = exists_box * target_classes # (BATCH, S, S, C)

    class_loss = mse(
        torch.flatten(pred_classes_obj, end_dim=-2), # Flatten (BATCH*S*S, C)
        torch.flatten(target_classes_obj, end_dim=-2)
    )

    # Loss Distance
    pred_distance_obj = exists_box * pred_distance # (BATCH, S, S, 1)
    target_distance_obj = exists_box * target_distance # (BATCH, S, S, 1)
    distance_loss = mse(
        torch.flatten(pred_distance_obj), # Flatten (BATCH*S*S, 1)
        torch.flatten(target_distance_obj) # Flatten (BATCH*S*S, 1)
    )

    total_loss = (
        lambda_coord * box_loss          # Localization loss
        + object_loss                         # Confidence loss (object present)
        + lambda_noobj * no_object_loss  # Confidence loss (no object present)
        + class_loss                          # Classification loss
        + distance_loss
    )

    # Average loss over the batch size
    batch_size = predictions.shape[0]
    total_loss = total_loss / batch_size

    return total_loss
