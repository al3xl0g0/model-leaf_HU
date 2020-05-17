from mrcnn.config import Config
import numpy as np

class LeafSegmentorConfig(Config):

    DN2RAD = np.asarray([0.059057, 0.192245, 0.594233, 1.198960, 1.871885, 2.034510, 2.075143])


    # Give the configuration a recognizable name
    NAME = "leaves"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1  # AZ 1  2   8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 shape (leaves)

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    #RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels
    #RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # AZ
    #RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # AZ train 2019_11_07
    #RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # AZ train 2019_11_08
    #RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # AZ train 2019_11_09
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # AZ train 2019_11_10


    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    #TRAIN_ROIS_PER_IMAGE = 62   #   32  16
    #TRAIN_ROIS_PER_IMAGE = 30   # AZ train 2019_11_09
    TRAIN_ROIS_PER_IMAGE = 62  # AZ train 2019_11_10

    # ROI_POSITIVE_RATIO = 66

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])  # ImageNet
    #MEAN_PIXEL = np.array([0.0, 0.0, 0.0])    # tst
    #MEAN_PIXEL = np.array([105.9, 103.1, 93.8]) # sample from training set - banana
