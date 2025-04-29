import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import *

class RadarFeatureNetwork(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        
        self.camera_count = constants.CAMERA_COUNT
        self.input_dim = constants.RADAR_INPUT_SIZE - 1 +  self.camera_count # Exclude the valid bit
        mlp_hidden_dim = 128
        self.output_dim = output_dim
        point_feature_dim = 128
        attention_hidden_dim = 128
        self.radar_mlp = nn.Sequential(
            nn.Linear(self.input_dim, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim), # LayerNorm often works well here
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, point_feature_dim),
            # nn.LayerNorm(output_dim),
            # nn.ReLU(inplace=True)
        )

        self.attention_projection  = nn.Linear(point_feature_dim, attention_hidden_dim)
        self.attention_scorer = nn.Linear(attention_hidden_dim, 1)

        self.final_projection = nn.Linear(point_feature_dim, output_dim) if point_feature_dim != output_dim else nn.Identity()
        self.final_activation = nn.ReLU(inplace=True) # Or another activation

    def forward(self, radar_data, camera_data):
        radar_input = radar_data[:, :, :constants.RADAR_INPUT_SIZE - 1]  # Exclude the valid bit
        mask = radar_data[:, :, constants.RADAR_INPUT_SIZE - 1]  # The valid bit

        # Expand camera data to match radar data shape
        camera_data_unsqueezed = camera_data.unsqueeze(1) # Shape: (32, 1, 5)
        camera_data_expanded = camera_data_unsqueezed.expand(-1, radar_input.shape[1], -1) # Shape: (32, 175, 5)

        radar_camera_input = torch.cat((radar_input, camera_data_expanded), dim=2)
        point_features = self.radar_mlp(radar_camera_input)
        
        # attention mechanism
        projected_features = torch.tanh(self.attention_projection(point_features))
        attention_logits = self.attention_scorer(projected_features).squeeze(-1)
        attention_logits.masked_fill_(mask == False, -float('inf'))
        attention_weights = F.softmax(attention_logits, dim=1)
        masked_attention_weights = attention_weights * mask.float()

        aggregated_features = torch.sum(masked_attention_weights.unsqueeze(-1) * point_features, dim=1)

        final_output = self.final_projection(aggregated_features)
        final_output = self.final_activation(final_output) # Apply final activation

        return final_output
    
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
        
        # unfreeze_indices = [12, 13]
        # print(f"Unfreezing CNN layers at indices: {unfreeze_indices}")
        # for i, layer in enumerate(self.car_recognition_model):
        #     if i in unfreeze_indices:
        #         print(f"  Unfreezing layer {i}: {layer}")
        #         for param in layer.parameters():
        #             param.requires_grad = True

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