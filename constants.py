class constants:
    # Constants for the image processing
    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224

    MAX_NUM_BBOXES = 2

    PRESCALE = False  

    LABEL_MAP =  {
            'SIZE_VEHICLE_M': 0,
            'SIZE_VEHICLE_XL': 0,
            'PEDESTRIAN': 1,
        } 
    NUM_OF_CLASSES = 2

    GRID_SIZE  = int(IMAGE_HEIGHT / 32)