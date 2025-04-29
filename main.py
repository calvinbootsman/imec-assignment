import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, MultiStepLR, SequentialLR
import torchvision.models as models
import torch.cuda.amp as amp
from torchvision.models import ResNet50_Weights
import os
import random
import json
import cv2
from dataset_loader import *
from constants import *
import time
import numpy as np
from cProfile import Profile
from pstats import SortKey, Stats
from car_recognition import *

from fusion import *

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
def save_model(model, time):
    """
    Save the model to the specified path.
    """
    model_save_path = os.path.join('models', f'model_{time}.pth')
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(5) #3
    
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = DatasetLoader('highway', num_boxes_per_cell=constants.MAX_NUM_BBOXES, grid_size=constants.GRID_SIZE)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])

    num_workers = 0
    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=num_workers, batch_size=32, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=num_workers, batch_size=32, shuffle=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=num_workers, batch_size=32, shuffle=False, pin_memory=True)

    car_model = CarRecorgnitionNetwork(num_classes=constants.NUM_OF_CLASSES, num_boxes=constants.MAX_NUM_BBOXES).to(device)
    car_model_path = 'models/model_036.pth'
    car_model.load_state_dict(torch.load(car_model_path))
    
    model = FusionNetwork(car_model, num_classes=constants.NUM_OF_CLASSES, num_boxes=constants.MAX_NUM_BBOXES).to(device)
    criterion = yolo_loss

    max_no_improvement = 500

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=max_no_improvement, min_lr=1e-4)

    scaler = torch.amp.GradScaler(enabled = (device.type == 'cuda'))
    peak_lr = 1e-2         # The LR achieved *after* warm-up (and start of phase 1)
    warmup_start_lr = 1e-3 # The LR at the very beginning of training
    warmup_epochs = 5
    phase1_epochs = 10
    phase2_epochs = 10
    phase3_epochs = 20

    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer = optim.SGD(model.parameters(), lr=peak_lr, momentum=0.9, weight_decay=0.0005)
    start_factor = warmup_start_lr / peak_lr
    warmup_scheduler = LinearLR(optimizer,
                                start_factor=start_factor,
                                end_factor=1.0, # End at the optimizer's base LR (peak_lr)
                                total_iters=warmup_epochs)
    decay_milestones = [
    warmup_epochs + phase1_epochs,
    warmup_epochs + phase1_epochs + phase2_epochs
    ]
    decay_gamma = 0.1 # Factor to decrease LR by (1e-2 -> 1e-3 -> 1e-4)
    decay_scheduler = MultiStepLR(optimizer,
                                milestones=decay_milestones,
                                gamma=decay_gamma)

    # 3. Combined Scheduler (Sequential)
    # Switch from warmup_scheduler to decay_scheduler *after* warmup_epochs.
    # The milestone indicates the epoch index where the *next* scheduler becomes active.
    sequential_milestones = [warmup_epochs]
    scheduler = SequentialLR(optimizer,
                            schedulers=[warmup_scheduler, decay_scheduler],
                            milestones=sequential_milestones)

    # --- Example Training Loop ---
    total_epochs = warmup_epochs + phase1_epochs + phase2_epochs + phase3_epochs
    for epoch in range(total_epochs):
        model.train()
        running_loss = 0.0
        i = 0
        # with Profile() as pr:
        #     pr.enable()
        start_time = time.time()
        for batch_data_item, batch_targets, batch_radar, batch_camera in train_loader:
            image = batch_data_item.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            batch_radar = batch_radar.to(device, non_blocking=True)
            batch_camera = batch_camera.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(image, batch_radar, batch_camera)
                image_target = batch_targets[..., :-1]  
                loss = criterion(outputs, image_target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:  # Print stats every 100 batches
                print(f'Epoch [{epoch+1}/{total_epochs}], Batch [{i+1}/{len(train_loader)}], Batch Loss: {(running_loss/100)}')
                running_loss = 0.0
            i += 1
        #     pr.disable()
        # stats = Stats(pr)
        # stats.sort_stats(SortKey.TIME)
        # stats.print_stats(10)  # Print the top 10 functions by time

        # Validation
        scheduler.step()
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for i, (batch_data_item, batch_targets, batch_radar, batch_camera) in enumerate(val_loader):
                image = batch_data_item.to(device, non_blocking=True)
                batch_targets = batch_targets.to(device, non_blocking=True)
                batch_radar = batch_radar.to(device, non_blocking=True)
                batch_camera = batch_camera.to(device, non_blocking=True)
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(image, batch_radar, batch_camera)
                    image_target = batch_targets[..., :-1]  
                    val_loss += loss.item()

        val_loss /= len(val_loader)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Epoch [{epoch+1}/{total_epochs}], Validation Loss: {val_loss}, Learning Rate: {scheduler.optimizer.param_groups[0]["lr"]}, Elapsed Time: {elapsed_time:.2f} seconds')
        save_model(model, start_time)
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
        

    