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

class DataItem:
    def __init__(self, image_tensor, radar_data, bounding_box_data=None, image_path=None):
        self.image_tensor = image_tensor 
        # self.radar_data = radar_data
        self.bounding_box_data = bounding_box_data
        self.image_path = image_path

    def __repr__(self):
        return f"DataItem(image_path={self.image_path}, radar_data_shape={self.radar_data.shape})"
    
    def save(self, output_dir, index):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        image_name = os.path.basename(self.image_path).replace('.jpg', '.json')
        # radar_data_path = os.path.join(output_dir, image_name)
        
        with open(radar_data_path, 'w') as f:
            json.dump(self.radar_data, f)

        # Save the image tensor as a numpy array
        image_tensor_path = os.path.join(output_dir, os.path.basename(self.image_path))
        np.save(image_tensor_path, self.image_tensor.numpy())

class DatasetGenerator(Dataset):
    def __init__(self, driving_style='highway', max_images=-1, num_boxes_per_cell=2, grid_size=7):
        self.data_path = f'train/{driving_style}'
        self.image_paths = []
        self.data_set = []
        self.max_images = max_images

        self.B = num_boxes_per_cell
        self.S = grid_size
        self.C = constants.NUM_OF_CLASSES

        self._index_images()

        self.label_map = {
            'SIZE_VEHICLE_M': 0,
            'SIZE_VEHICLE_XL': 0,
            'PEDESTRIAN': 1,
        }
    def generate(self):
        output_dir = 'processed_data'
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'image_paths.txt'), 'w') as f:
            for path in self.image_paths:
                f.write(f"{path}\n")
        for i, image_path in enumerate(self.image_paths):
            try:
                image_name = os.path.basename(image_path)
                scene = image_path.split('/')[-5]
                camera = image_path.split('/')[-2]
                file_number = image_name.replace('.jpg', '').replace(camera, '').replace('_', '')
                
                image_tensor, original_size = self._image_loader(image_path)
                target_tensor = self._bounding_box_loader(scene, camera, image_name, original_size)                   
                # radar_data = self._radar_loader(self.data_path, scene, file_number)
                # data_item = DataItem(image_tensor, radar_data, bounding_box_data, image_path=image_path)
                processed_data = {
                    'image': image_tensor,
                    'target': target_tensor
                    # Optional: 'radar': radar_tensor
                }
                # Save the processed data
                save_path = os.path.join(output_dir, f'sample_{i:06d}.pth')
                torch.save(processed_data, save_path)
                if i % 500 == 0:
                    print(f"Processed {i} images. Saved to {save_path}.", end='\r')
            except Exception as e:
                print(f"Error: {e}")
                continue

    def _image_loader(self, image_path: str):
        if constants.PRESCALE == False:
            original_image = cv2.imread(image_path)
            original_size = original_image.shape[:2]  # (height, width)
            image = cv2.resize(original_image, (constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT))

            # Convert the image to Torch tensor 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            new_path = image_path.replace(self.data_path, 'train/scaled_images')
            image = cv2.imread(new_path)
            # original_size = (constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH)
            if  "B_MIDRANGECAM_C" in image_path:
                original_size = (1920, 1216)
            else:
                original_size = (704, 1280)
        image = image.astype('float32') / 255.0
        transform = transforms.Compose([transforms.ToTensor()])
        tensor = transform(image)

        return tensor, original_size 

    def _bounding_box_loader(self, scene: str, camera: str, image_name: str, original_size: tuple):
        """
        Loads bounding box data from the specified path.
        """
        box_file = image_name.replace('.jpg', '.json')
        box_data_path = f'{self.data_path}/{scene}/dynamic/box/2d/{camera}/{box_file}'

        target_tensor = torch.zeros(self.S, self.S, self.B * 5 + self.C, dtype=torch.float32)

        if not os.path.exists(box_data_path):
            return target_tensor
        
        original_height, original_width = original_size

        with open(box_data_path, 'r') as box_file_handle:
            try:
                data = json.load(box_file_handle)
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON in {box_data_path}. Skipping boxes.")
                return target_tensor
            
            boxes_data = data.get('CapturedObjects', [])
            for box_json in boxes_data:
                object_type = box_json.get('ObjectType')
                if object_type not in self.label_map:
                    continue

                coord_keys = ['BoundingBox2D X1', 'BoundingBox2D Y1', 'BoundingBox2D X2', 'BoundingBox2D Y2']
                if not all(k in box_json and box_json[k] is not None for k in coord_keys):
                    # print(f"Warning: Missing coordinate data in {box_data_path} for object {object_type}. Skipping.")
                    continue

                x1 = float(box_json['BoundingBox2D X1']) / original_width
                y1 = float(box_json['BoundingBox2D Y1']) / original_height
                x2 = float(box_json['BoundingBox2D X2']) / original_width
                y2 = float(box_json['BoundingBox2D Y2']) / original_height

                x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], 0.0, 1.0)

                x_center_norm = (x1 + x2) / 2
                y_center_norm = (y1 + y2) / 2
                width_norm = abs(x2 - x1)
                height_norm = abs(y2 - y1)

                if width_norm <= 0 or height_norm <= 0:
                    continue
                
                x_cell_float = x_center_norm * self.S
                y_cell_float = y_center_norm * self.S
                grid_j = int(x_cell_float) # Column index
                grid_i = int(y_cell_float) # Row index
                
                x_rel_cell = x_cell_float - grid_j
                y_rel_cell = y_cell_float - grid_i

                class_id = self.label_map[object_type] 

                if target_tensor[grid_i, grid_j, 4] == 0:
                    # Assign Confidence = 1 to the first box slot
                    target_tensor[grid_i, grid_j, 4] = 1.0

                    # Assign Coordinates (rel x,y; image w,h) to the first box slot
                    target_tensor[grid_i, grid_j, 0] = x_rel_cell
                    target_tensor[grid_i, grid_j, 1] = y_rel_cell
                    target_tensor[grid_i, grid_j, 2] = width_norm
                    target_tensor[grid_i, grid_j, 3] = height_norm

                    # Assign Class probabilities (one-hot)
                    class_vector_start_index = self.B * 5
                    target_tensor[grid_i, grid_j, class_vector_start_index + class_id] = 1.0
        return target_tensor 
    
    def _index_images(self):
        """
        Indexes images in the dataset.
        """
        self.image_paths = []
        all_image_paths = []
        correct_image_paths = []
        for scene_folder in os.listdir(self.data_path):
            for camera_folder in os.listdir(os.path.join(self.data_path, scene_folder, 'sensor', 'camera')):
                if camera_folder == 'sync_frame2host.json':
                    continue
                
                image_names = [img for img in os.listdir(os.path.join(self.data_path, scene_folder, 'sensor', 'camera', camera_folder)) if img.endswith('.jpg')]
                image_paths = [os.path.join(self.data_path, scene_folder, 'sensor', 'camera', camera_folder, img).replace('\\', '/') 
                               for img in image_names]
                
                all_image_paths.extend(image_paths)
        
        # check if the file exists in the radar data folder
        for paths in all_image_paths:
            scene = paths.split('/')[-5]

            camera = paths.split('/')[-2]
            image_name = os.path.basename(paths)
            file_number = image_name.replace('.jpg', '').replace(camera, '').replace('_', '')
            
            front_radar_file = f'F_LRR_C_{file_number}.json'
            back_radar_file = f'B_LRR_C_{file_number}.json'
            front_radar_full_path = f'{self.data_path}/{scene}/sensor/radar/F_LRR_C/{front_radar_file}'
            back_radar_data_full_path = f'{self.data_path}/{scene}/sensor/radar/B_LRR_C/{back_radar_file}'
            
            if os.path.exists(front_radar_full_path) or os.path.exists(back_radar_data_full_path):
                correct_image_paths.append(paths)

        if len(correct_image_paths) > self.max_images and self.max_images > 0:
            correct_image_paths = random.sample(correct_image_paths, self.max_images)
        self.image_paths = correct_image_paths

if __name__ == "__main__":
    dataset_generator = DatasetGenerator(driving_style='highway', num_boxes_per_cell=2, grid_size=7)
    dataset_generator.generate()
    print(f"Generated {len(dataset_generator.image_paths)} samples.")