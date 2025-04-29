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

def save_model(model, time):
    """
    Save the model to the specified path.
    """
    model_save_path = os.path.join('obj_models', f'obj_model_{time}.pth')
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(5) #3
    
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = DatasetLoader('highway', num_boxes_per_cell=constants.MAX_NUM_BBOXES, grid_size=constants.GRID_SIZE)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    num_workers = 8
    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=num_workers, batch_size=32, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=num_workers, batch_size=32, shuffle=False, pin_memory=True)
    # test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=num_workers, batch_size=32, shuffle=False, pin_memory=True)

    car_model = CarRecorgnitionNetwork(num_classes=constants.NUM_OF_CLASSES, num_boxes=constants.MAX_NUM_BBOXES).to(device)
    # car_model_path = 'models/model_036.pth'
    # car_model.load_state_dict(torch.load(car_model_path))
    
    # model = FusionNetwork(car_model, num_classes=constants.NUM_OF_CLASSES, num_boxes=constants.MAX_NUM_BBOXES).to(device)
    model = car_model
    criterion = yolo_loss

    max_no_improvement = 500

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=max_no_improvement, min_lr=1e-4)

    scaler = torch.amp.GradScaler(enabled = (device.type == 'cuda'))
    peak_lr = 1e-2         # The LR achieved *after* warm-up (and start of phase 1)
    warmup_start_lr = 1e-3 # The LR at the very beginning of training
    warmup_epochs = 5
    phase1_epochs = 5
    phase2_epochs = 5
    phase3_epochs = 10

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
    training_start_time = time.time()
    # --- Example Training Loop ---
    total_epochs = warmup_epochs + phase1_epochs + phase2_epochs + phase3_epochs
    for epoch in range(total_epochs):
        model.train()
        running_loss = 0.0
        i = 0

        start_time = time.time()
        for batch_data_item, batch_targets, batch_radar, batch_camera in train_loader:
            image = batch_data_item.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            batch_radar = batch_radar.to(device, non_blocking=True)
            batch_camera = batch_camera.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                # outputs = model(image, batch_radar, batch_camera)
                outputs = model(image)
                image_target = batch_targets[..., :-1]  
                loss = criterion(outputs, image_target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:  # Print stats every 100 batches
                print(f'Epoch [{epoch+1}/{total_epochs}], Batch [{i+1}/{len(train_loader)}], Batch Loss: {(running_loss/100)}')
                running_loss = 0.0
            i += 1

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
                    # outputs = model(image, batch_radar, batch_camera)
                    outputs = model(image)
                    image_target = batch_targets[..., :-1]  
                    val_loss += loss.item()

        val_loss /= len(val_loader)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Epoch [{epoch+1}/{total_epochs}], Validation Loss: {val_loss}, Learning Rate: {scheduler.optimizer.param_groups[0]["lr"]}, Elapsed Time: {elapsed_time:.2f} seconds')
        save_model(model, training_start_time)
