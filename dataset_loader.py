import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os 
import random
import json
import cv2
from constants import *

class DataItem:
    def __init__(self, image, radar_data, bounding_box, image_path=None):
        self.image = image
        self.radar_data = radar_data
        self.bounding_box = bounding_box
        self.image_path = image_path

    def to_input(self):
        return self.image   

class BoundingBoxDataItem:
    def __init__(self, x1, y1, x2, y2, label):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.label = label

        x1_f, y1_f, x2_f, y2_f = float(x1), float(y1), float(x2), float(y2)
        self.x_center = (x1_f + x2_f) / 2
        self.y_center = (y1_f + y2_f) / 2
        self.width = x2_f - x1_f
        self.height = y2_f - y1_f

class RadarDataItem:
    def __init__(self, azimuth, range, rcs):
        self.azimuth = azimuth
        self.range = range
        self.rcs = rcs

class DatasetLoader:
    def __init__(self, device, driving_style='highway', downsample=1, max_images=-1, num_bounding_boxes=1):
        self.device = device
        self.data_path = f'train/{driving_style}'
        self.image_paths = []
        self.data_set = []
        self.downsample = downsample
        self.max_images = max_images
        self.num_bounding_boxes = num_bounding_boxes

        self._index_images()

        self.label_map = {
            'SIZE_VEHICLE_M': 0,
            'SIZE_VEHICLE_XL': 1,
            'PEDESTRIAN': 2,
            'TWO_WHEEL_WITHOUT_RIDER': 3,
            'RIDER': 4,
            'NONE': 5,
        } 
        
    def __getitem__(self, index):
        if index >= len(self.image_paths):
            raise IndexError("Index out of range")
        image_path = self.image_paths[index]
        while True:
            try:
                image_name = os.path.basename(image_path)
                scene = image_path.split('/')[-5]
                camera = image_path.split('/')[-2]
                file_number = image_name.replace('.jpg', '').replace(camera, '').replace('_', '')
                
                image = self._image_loader(image_path, scene, camera)
                bounding_box_data, target = self._bounding_box_loader(scene, camera, image_name)                   
                radar_data = self._radar_loader(self.data_path, scene, file_number)
                data_item = DataItem(image, radar_data, bounding_box_data, image_path=image_path)

                return data_item.to_input(), target
                
            except Exception as e:
                print(f"Error: {e}")
                continue

            
    def __len__(self):
        return len(self.image_paths)
    
    # def get_random_pictures(self, total_pictures: int):
    #     items = []
    #     for _ in range(total_pictures):
    #         while True:
    #             try:
    #                 image_path = self.image_paths.pop(random.randrange(len(self.image_paths)))

    #                 image_name = os.path.basename(image_path)
    #                 scene = image_path.split('/')[-5]
    #                 camera = image_path.split('/')[-2]
    #                 file_number = image_name.replace('.jpg', '').replace(camera, '').replace('_', '')
                    
    #                 original_image = cv2.imread(image_path)
    #                 original_size = original_image.shape
    #                 image = cv2.resize(original_image, (original_size[1] // downsample, original_size[0] // downsample))
    #                 camera_data = self._bounding_box_loader(scene, camera, image_name)                   
    #                 radar_data = self._radar_loader(self.data_path, scene, file_number)
    #                 data_item = DataItem(image, radar_data, camera_data)
    #                 items.append(data_item)
                    
    #             except Exception as e:
    #                 print(f"Error: {e}")
    #                 continue
    #             break

    #     self.data_set = items
    #     return items
    
    def _image_loader(self, image_path: str, image_width: int, image_height: int):
        original_image = cv2.imread(image_path)
        image = cv2.resize(original_image, (constants.IMAGE_WIDTH // self.downsample, constants.IMAGE_HEIGHT // self.downsample))
        # image = cv2.resize(original_image, (original_size[1] // self.downsample, original_size[0] // self.downsample))

        # Convert the image to Torch tensor 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([transforms.ToTensor()])
        tensor = transform(image).to(self.device)

        return tensor

    def _bounding_box_loader(self, scene: str, camera: str, image_name: str):
        """
        Loads bounding box data from the specified path.
        """
        box_file = image_name.replace('.jpg', '.json')
        box_data = f'{self.data_path}/{scene}/dynamic/box/2d/{camera}/{box_file}'
        target_list = []
        bounding_boxes = []
        if not os.path.exists(box_data):
            # raise FileNotFoundError(f"Bounding box data file not found: {box_data}")
            for _ in range(self.num_bounding_boxes):
                target_tensor = torch.tensor([-1, -1, -1, -1, -1, 0.0], dtype=torch.float32, device=self.device)
                target_list.append(target_tensor)
            return bounding_boxes, target_list
        
        with open(box_data, 'r') as box_file_handle:
            data = json.load(box_file_handle)
            boxes = data.get('CapturedObjects', [])
            bounding_boxes_coords = []
            bounding_boxes_labels = []
            for box in boxes:
                # Check if the box is a dictionary and contains the required keys
                if box.get('ObjectType') in self.label_map \
                    and isinstance(box, dict) \
                    and all(k in box and box[k] is not None for k in ['BoundingBox2D X1', 'BoundingBox2D Y1', 'BoundingBox2D X2', 'BoundingBox2D Y2', 'ObjectType']):
                    bounding_boxes_coords.append([
                        max(0.0, box['BoundingBox2D X1'] / self.downsample),
                        max(0.0, box['BoundingBox2D Y1'] / self.downsample),
                        max(0.0, box['BoundingBox2D X2'] / self.downsample),
                        max(0.0, box['BoundingBox2D Y2'] / self.downsample)
                    ])
                    bounding_boxes_labels.append(box['ObjectType'])
            bounding_boxes = []
            if bounding_boxes_coords:
                bounding_boxes_tensor = torch.tensor(bounding_boxes_coords, dtype=torch.float32, device=self.device)
            else:
                bounding_boxes_tensor = torch.empty((0, 4), dtype=torch.float32, device=self.device)

            for i in range(len(bounding_boxes_tensor)):
                x1, y1, x2, y2 = bounding_boxes_tensor[i]
                bounding_boxes.append(BoundingBoxDataItem(x1.item(), y1.item(), x2.item(), y2.item(), bounding_boxes_labels[i]))

            
            for i in range(len(bounding_boxes)):
                x_center, y_center, width, height = bounding_boxes[i].x_center, bounding_boxes[i].y_center, bounding_boxes[i].width, bounding_boxes[i].height
                label = self.label_map.get(bounding_boxes_labels[i], -1)
                target_list.append([label, x_center, y_center, width, height, 1.0]) # 1.0 is the confidence score

            if len(target_list) > self.num_bounding_boxes:
                print(f"Warning: More bounding boxes than expected. Found {len(target_list)} but expected {self.num_bounding_boxes}.")

            for i in range(self.num_bounding_boxes -  len(target_list)):
                target_list.append(torch.tensor([-1, -1, -1, -1, -1, 0.0], dtype=torch.float32, device=self.device))

            return bounding_boxes, target_list  
        
    def _radar_loader(self, data_path: str, folder: str, file_number: str):
        """
        Loads radar data from the specified path.
        """
        front_radar_file = f'F_LRR_C_{file_number}.json'
        back_radar_file = f'B_LRR_C_{file_number}.json'
        front_radar_full_path = f'{data_path}/{folder}/sensor/radar/F_LRR_C/{front_radar_file}'
        back_radar_data_full_path = f'{data_path}/{folder}/sensor/radar/B_LRR_C/{back_radar_file}'
        datasets_paths = [front_radar_full_path, back_radar_data_full_path]

        radar_data = []
        for path in datasets_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Radar data file not found: {path}")
            
            with open(path, 'r') as radar_file:
                data = json.load(radar_file)
                targets = data.get('targets', [])
                for target in targets:
                     if isinstance(target, dict) and all(k in target and target[k] is not None for k in ['azimuth', 'range', 'rcs']):
                         radar_data.append([
                             target['azimuth'],
                             target['range'],
                             target['rcs']
                         ])

        if radar_data:
            radar_tensor = torch.tensor(radar_data, dtype=torch.float32, device=self.device)
        else:
            radar_tensor = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        return radar_tensor
    
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
    dataset_loader = DatasetLoader('highway')

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset_loader, [0.8, 0.1, 0.1])
    
    # items = dataset_loader.get_random_pictures(1)
    image_index = train_dataset.indices[0]
    image_path = dataset_loader.image_paths[image_index]
    image = cv2.imread(image_path)

    # radar_data = data_entry[0].radar_data

    _, bounding_boxes = train_dataset[0]
    
    # Display the image with bounding boxes
    if bounding_boxes is not None:
        for box in bounding_boxes:
            x_center, y_center = int(box[1]), int(box[2])
            width, height = int(box[3]), int(box[4])
            x1, y1 = x_center - width // 2, y_center - height // 2
            x2, y2 = x_center + width // 2, y_center + height // 2
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        print("No bounding boxes found.")
    cv2.imshow('image', image)
    cv2.waitKey(0)