class constants:
    # Constants for the image processing
    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224

    MAX_NUM_BBOXES = 2

    PRESCALE = True  

    LABEL_MAP =  {
            'SIZE_VEHICLE_M': 0,
            'SIZE_VEHICLE_XL': 0,
            'PEDESTRIAN': 1,
        } 
    NUM_OF_CLASSES = 2

    GRID_SIZE  = int(IMAGE_HEIGHT / 32)
    
    MAX_RADAR_DISTANCE = 282
    MAX_RADAR_POINTS = 175
    MAX_AZIMUTH = 0.3
    MAX_RCS = 1000.0
    MAX_NOISE = 126
    RADAR_INPUT_SIZE = 6 # azimuth, rcs, noise, range, front/back, valid

    CAMERA_COUNT = 5