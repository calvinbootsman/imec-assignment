import cv2
import numpy as np
import torch
import torch.nn as nn
from constants import *
import os
from dataset_loader import * 
from car_recognition import *
import random
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
                    
                    # Get the distance estimate (if applicable)
                    distance_estimate = tensor_data[i, j, box_offset + 5] if (box_offset + 5) < tensor_data.shape[2] else None
                    if distance_estimate is not None:
                        distance_estimate = distance_estimate.item()
                        label += f" (Dist: {distance_estimate:.2f})"
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

def load_latest_model(models_folder: str):
    """
    Loads the latest PyTorch model from the specified folder.

    Args:
        models_folder (str): Path to the folder containing model files.

    Returns:
        torch.nn.Module: The loaded PyTorch model.
    """
    # Get all files in the models folder
    model_files = [f for f in os.listdir(models_folder) if f.endswith('.pt') or f.endswith('.pth')]
    
    if not model_files:
        raise FileNotFoundError("No model files found in the specified folder.")

    # Sort files by modification time (newest first)
    model_files.sort(key=lambda f: os.path.getmtime(os.path.join(models_folder, f)), reverse=True)
    
    # Load the latest model
    latest_model_path = os.path.join(models_folder, model_files[0])
    car_model = CarRecorgnitionNetwork(num_classes=constants.NUM_OF_CLASSES, num_boxes=constants.MAX_NUM_BBOXES).to(device)
    car_model_path = 'obj_models/obj_model_1745947072.330789.pth'
    car_model.load_state_dict(torch.load(car_model_path))
    
    print(f"Loading model from: {latest_model_path}")
    model = FusionNetwork(car_recognition_model=car_model, num_classes=constants.NUM_OF_CLASSES, num_boxes=constants.MAX_NUM_BBOXES).to(device)
    model.load_state_dict(torch.load(latest_model_path))
    return model

if __name__ == "__main__":
    random.seed(40)
    torch.manual_seed(5) #3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models_folder = "models"
    latest_model = load_latest_model(models_folder)
    latest_model.eval()

    dataset = DatasetLoader('highway',max_images=10_000, num_boxes_per_cell=constants.MAX_NUM_BBOXES, grid_size=constants.GRID_SIZE)
    images = []
    for i in range(5):
        random_index = random.randint(0, len(dataset) - 1)
        random_sample = dataset[random_index]
        image_tensor, targets_tensor, radar, camera = random_sample

        image = dataset.get_original_image(random_index)
        image_tensor = image_tensor.unsqueeze(0).to(device)
        radar = radar.unsqueeze(0).to(device)
        camera = camera.unsqueeze(0).to(device)
        
        outputs = latest_model(image_tensor, radar, camera)
        output_image = draw_bounding_boxes_yolo(image, outputs[0], constants.GRID_SIZE, constants.MAX_NUM_BBOXES, constants.NUM_OF_CLASSES, threshold=0.8)
        # images.append(output_image)
        cv2.imshow("YOLO Output", output_image)
        cv2.waitKey(0)
        # cv2.destroyAllWindows()