import torch
import torch.nn as nn
from constants import *

class RadarFeatureNetwork(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        
        self.camera_count = constants.CAMERA_COUNT
        self.input_dim = constants.RADAR_INPUT_SIZE - 1 +  self.camera_count # Exclude the valid bit
        hidden_dim = 128
        self.output_dim = output_dim

        self.radar_mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # LayerNorm often works well here
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, radar_data, camera_data):
        radar_input = radar_data[:, :, :constants.RADAR_INPUT_SIZE - 1]  # Exclude the valid bit
        mask = radar_data[:, :, constants.RADAR_INPUT_SIZE - 1]  # The valid bit

        # Expand camera data to match radar data shape
        camera_data_unsqueezed = camera_data.unsqueeze(1) # Shape: (32, 1, 5)
        camera_data_expanded = camera_data_unsqueezed.expand(-1, radar_input.shape[1], -1) # Shape: (32, 175, 5)

        radar_camera_input = torch.cat((radar_input, camera_data_expanded), dim=2)
        point_features = self.radar_mlp(radar_camera_input)

        masked_features = point_features * mask.unsqueeze(-1).float()  # Apply the mask to the features

        # Masked Average Pooling
        summed_features = torch.sum(masked_features, dim=1)  # Sum over the radar points
        num_valid_points = torch.sum(mask, dim=1) + 1e-6 # Shape: [B]
        num_valid_points_for_div = num_valid_points.unsqueeze(1) # Shape: [B, 1]
        averaged_features = summed_features / num_valid_points_for_div  # Average over the valid points
        return averaged_features
    
class FusionNetwork(nn.Module):
    def __init__(self, car_recognition_model, num_classes=1, num_boxes=1, hidden_dim=512):
        super().__init__()
        
        self.output_features_per_cell = (num_classes* 5) + num_classes + 1 # 4 coordinates + 1 confidence score + distance estimate
        self.output_features_per_box = (num_boxes * 5) + num_classes
        self.input_dims = (3, constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH)

        self.car_recognition_model = car_recognition_model.cnn
        car_output_dim = car_recognition_model.cnn_output_dim

        for param in self.car_recognition_model.parameters():
                param.requires_grad = False
        self.car_recognition_model.eval() 

        self.radar_feature_network = RadarFeatureNetwork()
        self.radar_output_dim = self.radar_feature_network.output_dim

        self.fusion_dim = car_output_dim + self.radar_output_dim
        self.dense_layers = nn.Sequential(
                nn.Linear(self.fusion_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1, inplace=True),      
                nn.Dropout(0.5),          
                nn.Linear(hidden_dim, int(constants.GRID_SIZE * constants.GRID_SIZE * self.output_features_per_cell)),
                nn.Sigmoid()
            )
        
        self.flatten = nn.Flatten()


    def forward(self, image, radar_data, camera_data):
        camera_features_map = self.car_recognition_model(image)
        camera_features = self.flatten(camera_features_map)
        radar_features = self.radar_feature_network(radar_data, camera_data)

        fused_features = torch.cat((camera_features, radar_features), dim=1)
        output = self.dense_layers(fused_features)
        output = output.view(-1, constants.GRID_SIZE, constants.GRID_SIZE, self.output_features_per_cell)
        return output