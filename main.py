import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import os
import random
import json
import cv2
from dataset_loader import *
from constants import *
import time
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, grid_size=16, num_classes=1, num_boxes=1):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.output_features_per_cell = (num_classes* 5) + num_classes# 4 coordinates + 1 confidence score
        self.output_features_per_box = (num_boxes * 5) + num_classes
        self.input_dims = (3, constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH)

        # Load pre-trained ResNet50
        base_model = models.resnet50(pretrained=True)

        # Remove the last fully connected layer
        base_model = nn.Sequential(*list(base_model.children())[:-2])

        # Freeze the base model
        for param in base_model.parameters():
            param.requires_grad = False

        NUM_FILTERS = 512
        cnn = nn.Sequential(
            # Based on the VGG16 architecture
            # https://arxiv.org/pdf/1409.1556.pdf
            nn.Conv2d(2048, NUM_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU(0.1),
            # nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(0.1, inplace=True),
        )

        flatten = nn.Flatten()

        dense_layers = nn.Sequential(
            nn.Linear(NUM_FILTERS * (constants.IMAGE_HEIGHT // 32) * (constants.IMAGE_WIDTH // 32), NUM_FILTERS),
            nn.BatchNorm1d(NUM_FILTERS),
            nn.LeakyReLU(0.1),
            
            nn.Linear(NUM_FILTERS, int(constants.GRID_SIZE * constants.GRID_SIZE * self.output_features_per_cell)),
            nn.Sigmoid()
        )

        # Combine all layers into a single sequential model
        self.model = nn.Sequential(
            base_model,
            cnn,
            flatten,
            dense_layers,
            nn.Unflatten(1, (constants.GRID_SIZE, constants.GRID_SIZE, self.output_features_per_cell))
        )
        # with torch.no_grad():
        #     dummy_input = torch.zeros(1, *self.input_dims) 
        #     cnn_out_shape = self.cnn(dummy_input).shape
        #     self.flattened_size = cnn_out_shape[1] * cnn_out_shape[2] * cnn_out_shape[3]

        # Fully connected layers
        self.detection_head  = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # Extra conv layers often help
            nn.LeakyReLU(0.1, inplace=True),
            # Final convolution maps features to the desired output shape per cell
            nn.Conv2d(512, self.output_features_per_cell, kernel_size=1, stride=1, padding=0)
        )

        # self.fc = nn.Sequential(
        #     nn.Flatten(), 
        #     nn.Linear(self.flattened_size, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(0.1), 
        #     nn.Linear(1024, self.num_boxes * self.output_features_per_box)
        # )
    def forward(self, x):
        # features = self.cnn(x)
        # predictions = self.detection_head(features)
        # # predictions = predictions.view(-1, self.num_boxes, self.output_features_per_box)
        # predictions = predictions.permute(0, 2, 3, 1)
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
    pred_confs = []
    B = constants.MAX_NUM_BBOXES
    S = constants.GRID_SIZE
    C = constants.NUM_OF_CLASSES
    mse = nn.MSELoss(reduction="sum")
    lambda_noobj = 0.5
    lambda_coord = 5
    for b in range(B):
        offset = b * 5
        pred_boxes.append(predictions[..., offset : offset+4])
        pred_confs.append(predictions[..., offset+4 : offset+5]) # Keep dim
    pred_classes = predictions[..., B*5 :]

    target_box = target[..., 0:4] # [x, y, w, h]
    target_conf = target[..., 4:5] # Confidence (objectness) - should be 1 if object present, 0 otherwise
    target_classes = target[..., B*5 :] # One-hot encoded classes
    
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
    pred_conf_obj = exists_box * pred_confs[0] # (BATCH, S, S, 1)
    target_conf_obj = exists_box * target_conf # Target is already 1 where object exists
    object_loss = mse(
            torch.flatten(pred_conf_obj), # Flatten (BATCH*S*S)
            torch.flatten(target_conf_obj)
    )

    # Loss No Object
    no_object_loss = 0.0
    no_exists_box = (1.0 - exists_box) # Mask for cells without objects
    for b in range(B):
            pred_conf_noobj = no_exists_box * pred_confs[b]
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

    total_loss = (
        lambda_coord * box_loss          # Localization loss
        + object_loss                         # Confidence loss (object present)
        + lambda_noobj * no_object_loss  # Confidence loss (no object present)
        + class_loss                          # Classification loss
    )

    # Average loss over the batch size
    batch_size = predictions.shape[0]
    total_loss = total_loss / batch_size

    return total_loss

def draw_bounding_boxes_yolo(image: np.ndarray,
                             tensor_data: torch.Tensor,
                             grid_size: int,
                             num_boxes: int,
                             num_classes: int,
                             class_names: list = constants.LABEL_MAP,
                             threshold: float = 0.5):
    """
    Draws bounding boxes on an image based on a YOLO-style output tensor.

    Args:
        image (np.ndarray): The image to draw on (should be in BGR format for cv2).
                            Make sure it has the original dimensions, not the resized network input size.
        tensor_data (torch.Tensor): The output tensor from the network or the target tensor.
                                     Shape should be (S, S, B*5 + C).
        grid_size (int): The grid size (S) used by the model/target tensor.
        num_boxes (int): The number of boxes predicted per grid cell (B).
        num_classes (int): The number of classes (C).
        class_names (list): A list of strings containing the names for each class index.
        threshold (float): The confidence threshold. Boxes with confidence below this
                           will not be drawn.

    Returns:
        np.ndarray: The image with bounding boxes drawn.
    """
    img_h, img_w, _ = image.shape
    S = grid_size
    B = num_boxes
    C = num_classes

    # Ensure tensor_data is on CPU for easier processing if it came from GPU
    tensor_data = tensor_data.cpu()

    # Iterate through each grid cell
    for i in range(S): # Row index (y-axis)
        for j in range(S): # Column index (x-axis)
            # Iterate through each predicted box within the cell
            for b in range(B):
                box_offset = b * 5
                # --- 1. Check Confidence Score ---
                confidence_index = box_offset + 4
                confidence = tensor_data[i, j, confidence_index]

                if confidence >= threshold:
                    # --- 2. Extract Box Coordinates ---
                    x_rel_cell = tensor_data[i, j, box_offset + 0]
                    y_rel_cell = tensor_data[i, j, box_offset + 1]
                    width_norm = tensor_data[i, j, box_offset + 2]
                    height_norm = tensor_data[i, j, box_offset + 3]

                    # --- 3. Convert to Image Coordinates ---
                    # Calculate absolute center (normalized 0-1)
                    # Recall: i is row (y), j is column (x)
                    x_center_norm = (j + x_rel_cell) / S
                    y_center_norm = (i + y_rel_cell) / S

                    # Convert to pixel coordinates
                    x_center_pix = x_center_norm * img_w
                    y_center_pix = y_center_norm * img_h
                    width_pix = width_norm * img_w
                    height_pix = height_norm * img_h

                    # Calculate corner coordinates (x1, y1, x2, y2)
                    x1 = int(x_center_pix - width_pix / 2)
                    y1 = int(y_center_pix - height_pix / 2)
                    x2 = int(x_center_pix + width_pix / 2)
                    y2 = int(y_center_pix + height_pix / 2)

                    # Clamp coordinates to image bounds
                    x1 = max(0, min(x1, img_w - 1))
                    y1 = max(0, min(y1, img_h - 1))
                    x2 = max(0, min(x2, img_w - 1))
                    y2 = max(0, min(y2, img_h - 1))

                    # --- 4. Extract Class Information ---
                    class_probs_start_index = B * 5
                    class_probs = tensor_data[i, j, class_probs_start_index : class_probs_start_index + C]
                    # Get the index and score of the most likely class
                    class_score, class_index = torch.max(class_probs, dim=-1)
                    class_index = class_index.item()
                    class_score = class_score.item()

                    # Get the class name
                    label = "Unknown"
                    for keys, value in class_names.items():
                        if class_index == value:
                            label = keys
                            break
                    

                    # --- 5. Draw Rectangle and Label ---
                    # Create label text with class, class score, and box confidence
                    label_text = f"{label}: {class_score:.2f} (Conf: {confidence:.2f})"
                    color = (0, 255, 0) # Green color for boxes

                    # Draw the rectangle
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                    # Draw the label text above the rectangle
                    (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    # Put background for text
                    cv2.rectangle(image, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1)
                    # Put text
                    cv2.putText(image, label_text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Black text

    return image

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(5) #3
    
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = DatasetLoader('highway',max_images=1_000, num_boxes_per_cell=constants.MAX_NUM_BBOXES, grid_size=constants.GRID_SIZE)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])

    num_workers = 7
    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=num_workers, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=num_workers, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=num_workers, batch_size=32, shuffle=False)

    model = NeuralNetwork(grid_size=constants.GRID_SIZE, num_classes=constants.NUM_OF_CLASSES, num_boxes=constants.MAX_NUM_BBOXES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = yolo_loss

    max_no_improvement = 50
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=max_no_improvement)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        i = 0
        for batch_data_item, batch_targets in train_loader:
            image = batch_data_item.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            outputs = model(image)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99: # Print stats every 100 batches
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Batch Loss: {(running_loss/100)}')
                running_loss = 0.0

            
            i += 1
        # Validation
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for i, (batch_data_item, batch_targets) in enumerate(val_loader):
                image = batch_data_item.to(device)
                batch_targets = batch_targets.to(device)
                outputs = model(image)
                loss = criterion(outputs, batch_targets)
                scheduler.step(loss)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss}, Learning Rate: {scheduler.optimizer.param_groups[0]["lr"]}')

    model.eval()
    random.seed(42)  # For reproducibility
    random_index = random.randint(0, len(dataset) - 1)
    random_sample = dataset[random_index]
    image_tensor, target = random_sample
    image = dataset.get_original_image(random_index)
    output_image = draw_bounding_boxes_yolo(image, target, constants.GRID_SIZE, constants.MAX_NUM_BBOXES, constants.NUM_OF_CLASSES)
    cv2.imshow('Output', output_image)
    cv2.waitKey(0)

    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension and move to device
    # image = dataset.get_original_image(random_index)
    outputs = model(image_tensor)
    output_image = draw_bounding_boxes_yolo(image, outputs[0], constants.GRID_SIZE, constants.MAX_NUM_BBOXES, constants.NUM_OF_CLASSES)
    cv2.imshow('Output', output_image)
    cv2.waitKey(0)
    # Save the model

    current_time = time.strftime("%Y%m%d-%H%M%S")
    model_save_path = os.path.join('models', f'model_{current_time}.pth')
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    # stop_time = time.time()
    # print(f"Total time taken: {stop_time - start_time:.2f} seconds")

        # val_loss = 0.0
        # with torch.no_grad():
        #     for i, (batch_data_item, batch_targets) in enumerate(val_loader):
        #         image = batch_data_item.to(device)
        #         batch_targets = batch_targets.to(device)
        #         outputs = model(image)
        #         loss = criterion(outputs, batch_targets)
        #         val_loss += loss.item()
        # val_loss /= len(val_loader)
        

    