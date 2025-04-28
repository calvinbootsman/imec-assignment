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

        self.camera_map = {
            'B_MIDRANGECAM_C': 0,
            'F_MIDLONGRANGECAM_CL': 1,
            'F_MIDLONGRANGECAM_CR': 2,
            'M_FISHEYE_L': 3,
            'M_FISHEYE_R': 4,
        }

    def generate(self):
        output_dir = 'processed_data_with_radar'
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'image_paths.txt'), 'w') as f:
            for path in self.image_paths:
                f.write(f"{path}\n")
        max_azimuth = 0
        max_rcs = 0
        max_noise = 0
        average_rcs = []
        max_targets = 0
        for i, image_path in enumerate(self.image_paths):
            try:
                image_name = os.path.basename(image_path)
                scene = image_path.split('/')[-5]
                camera = image_path.split('/')[-2]
                file_number = image_name.replace('.jpg', '').replace(camera, '').replace('_', '')
                
                image_tensor, original_size = self._image_loader(image_path)
                target_tensor = self._bounding_box_loader(scene, camera, image_name, original_size)                   
                radar_tensor = self._radar_signal_loader(self.data_path, scene, file_number)
                # size = radar_tensor.shape[0]
                # if size > max_targets:
                    # max_targets = size
                # if targets > max_targets:
                #     max_targets = targets
                camera_tensor = self._camera_loader(camera)
                # max_azimuth = max(max_azimuth, np.abs(azimuth))
                # max_rcs = max(max_rcs, rcs)
                # max_noise = max(max_noise, noise)
                # average_rcs.append(rcs)
                processed_data = {
                    'image': image_tensor,
                    'target': target_tensor,
                    'radar': radar_tensor,
                    'camera': camera_tensor,
                }
                # Save the processed data
                save_path = os.path.join(output_dir, f'sample_{i:06d}.pth')
                torch.save(processed_data, save_path)
                if i % 500 == 0:
                    print(f"Processed {i} images. Saved to {save_path}.", end='\r')
                    
            except Exception as e:
                print(f"Error: {e}")
                continue
        # print(f"\nMax Targets: {max_targets}")
        # print(f"\nMax Azimuth: {max_azimuth}, Max RCS: {max_rcs}, Max Noise: {max_noise}, Average RCS: {np.median(average_rcs)}")

    def _camera_loader(self, camera: str):
        output = [0] * 5
        output[self.camera_map[camera]] = 1
        return torch.tensor(output, dtype=torch.float32)
    
    def _radar_signal_loader(self, data_path: str, scene: str, file_number: str):
        front_radar_file = f'F_LRR_C_{file_number}.json'
        back_radar_file = f'B_LRR_C_{file_number}.json'
        front_radar_full_path = f'{data_path}/{scene}/sensor/radar/F_LRR_C/{front_radar_file}'
        back_radar_data_full_path = f'{data_path}/{scene}/sensor/radar/B_LRR_C/{back_radar_file}'
        datasets_paths = [front_radar_full_path, back_radar_data_full_path]
        radar_data = []
        max_targets = 0
        for i, path in enumerate(datasets_paths):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Radar data file not found: {path}")
            
            with open(path, 'r') as radar_file:
                data = json.load(radar_file)
                targets = data.get('targets', [])
                if len(targets) > max_targets:
                    max_targets = len(targets)
                for target in targets:
                     if isinstance(target, dict) and all(k in target and target[k] is not None for k in ['azimuth', 'range', 'rcs']):
                        radar_data.append([
                            np.clip(target['azimuth'] / constants.MAX_AZIMUTH, -1.0, 1.0), # Normalize and clip azimuth
                            np.clip(target['rcs'] / constants.MAX_RCS, 0.0, 1.0), # Normalize and clip RCS
                            np.clip(target['noise'] / constants.MAX_NOISE, 0.0, 1.0), # Normalize and clip noise
                            float(i), # 0 for front, 1 for back
                            1.0 # for valid targets
                        ])

        if len(radar_data) < constants.MAX_RADAR_POINTS:
            radar_data += [[0, 0, 0, 0, 0]] * (constants.MAX_RADAR_POINTS - len(radar_data))
            radar_tensor = torch.tensor(radar_data, dtype=torch.float32)
        else:
            radar_tensor = torch.tensor(radar_data, dtype=torch.float32)

        return radar_tensor
    
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
        distance_feature_dim = 1
        target_tensor = torch.zeros(self.S, self.S, self.B * 5 + self.C + distance_feature_dim, dtype=torch.float32)
        distance_index = self.B * 5 + self.C

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
                    # Assign Coordinates (rel x,y; image w,h) to the first box slot
                    target_tensor[grid_i, grid_j, 0] = x_rel_cell
                    target_tensor[grid_i, grid_j, 1] = y_rel_cell
                    target_tensor[grid_i, grid_j, 2] = width_norm
                    target_tensor[grid_i, grid_j, 3] = height_norm

                    # Assign Confidence = 1 to the first box slot
                    target_tensor[grid_i, grid_j, 4] = 1.0
                    
                    # Assign Class probabilities (one-hot)
                    class_vector_start_index = self.B * 5                    
                    target_tensor[grid_i, grid_j, class_vector_start_index + class_id] = 1.0

                    # Assign Distance Feature
                    distance = self._distance_loader(scene, camera, image_name, box_json['ObjectId'])
                    target_tensor[grid_i, grid_j, distance_index] = distance

        return target_tensor 
    
    def _distance_loader(self, scene: str, camera: str, image_name: str, object_id: int):
        box_3d_file = image_name.replace(camera, 'frame').replace('.jpg', '.json') 
        box_data_path_3d = f'{self.data_path}/{scene}/dynamic/box/3d_body/{box_3d_file}'
        fault_tensor = torch.zeros(1, dtype=torch.float32)
        if not os.path.exists(box_data_path_3d):
            print(f"Warning: 3D box data file not found: {box_data_path_3d}.")
            return fault_tensor
        
        with open(box_data_path_3d, 'r') as box_file_handle:
            try:
                data = json.load(box_file_handle)
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON in {box_data_path_3d}. Skipping boxes.")
                return fault_tensor
            
            boxes_data = data.get('CapturedObjects', [])
            for box_json in boxes_data:
                if box_json.get('ObjectId') == object_id:
                    x = box_json["BoundingBox3D Origin X"]
                    y = box_json["BoundingBox3D Origin Y"]
                    z = box_json["BoundingBox3D Origin Z"]
                    return torch.sqrt(torch.tensor(x**2 + y**2 + z**2, dtype=torch.float32))
                
        return fault_tensor
    
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
            
            if os.path.exists(front_radar_full_path) and os.path.exists(back_radar_data_full_path):
                correct_image_paths.append(paths)

        if len(correct_image_paths) > self.max_images and self.max_images > 0:
            correct_image_paths = random.sample(correct_image_paths, self.max_images)
        self.image_paths = correct_image_paths

if __name__ == "__main__":
    dataset_generator = DatasetGenerator(driving_style='highway', num_boxes_per_cell=2, grid_size=7)
    dataset_generator.generate()
    print(f"Generated {len(dataset_generator.image_paths)} samples.")