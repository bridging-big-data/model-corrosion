"""
Mask R-CNN

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet
"""

import os

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2,3" # specify which GPU(s) to be used

import sys
import json
import datetime
import numpy as np
import skimage.draw

# test ..
import scipy
from scipy import misc
import matplotlib.pyplot as plt
import imgaug

import cv2
import glob

from keras.callbacks import TensorBoard

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################


class Configuration(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "corrosion"
    
    # Can override..? yes. make sure for batch_size!
    GPU_COUNT = 2

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # Background + target)

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 50 #50
    
    LEARNING_RATE = 0.0001

    # Minimum probability value to accept a detected instance ROIs below this threshold are skipped
    # Skip detections with < 80% confidence
    #DETECTION_MIN_CONFIDENCE = 0.8

    # Non-maximum suppression threshold for detection
    #DETECTION_NMS_THRESHOLD = 0.3

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512 
    IMAGE_MAX_DIM = 512

    # Maximum number of ground truth instances to use in one image
    #MAX_GT_INSTANCES = 30 #5
    
    # Max number of final detections
    #DETECTION_MAX_INSTANCES = 30 #5

    # Length of square anchor side in pixels
    #RPN_ANCHOR_SCALES = (16, 64, 128, 256, 512) #(16, 64, 128, 160, 192) #(10, 20, 40, 80, 160)  #(32, 64, 128, 256, 512) #(32, 64, 360, 640) #(16, 32, 64, 128, 256) #(10,20,40,80,160)
    
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #TRAIN_BN = False  # Defaulting to False since batch size is often small
    
    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    #USE_MINI_MASK = False

    # Weight decay regularization
    WEIGHT_DECAY = 0.001 #0.0001
    
    # Mask branch (default x2)
    #POOL_SIZE = 14
    #MASK_POOL_SIZE = 28
    #MASK_SHAPE = [56, 56]
    
    # For soft NMS (overwrite here)
    SCORE_THRESHOLD = 0.1
    SOFT_NMS_SIGMA = 0.5
    
    #TRAIN_ROIS_PER_IMAGE = 200
    
    
############################################################
#  Dataset
############################################################

class Datasets(utils.Dataset):
    def load_data(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes.
        self.add_class("corrosion", 1, "fair")
        self.add_class("corrosion", 2, "poor")
        self.add_class("corrosion", 3, "severe")

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        
        dataset_dir = os.path.join(dataset_dir, subset)        
        image_ids = next(os.walk(dataset_dir))[2]
        
        for image_id in image_ids:
            if image_id.endswith(".jpeg"):

                self.add_image(
                    "corrosion",
                    image_id=image_id,
                    path=os.path.join(dataset_dir, image_id)
                    )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(info['path']))
        filename = os.path.basename(info['path'])
        
        # Read mask files from numpy array
        mask = []
        for f in next(os.walk(mask_dir))[2]:
            if f == (os.path.splitext(filename)[0]+".npy"):
                # mask = np.load(os.path.join(mask_dir, f)).astype(np.bool)
                mask = np.load(os.path.join(mask_dir, f))
        
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        #return mask, np.ones([mask.shape[-1]], dtype=np.int32)
        
        class_ids = []
        for i in range(mask.shape[-1]):
            class_id = np.unique(mask[:,:,i])
            class_ids.append(class_id[1])
        
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask.astype(np.bool), class_ids
        
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "crack":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
            

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = Datasets()
    dataset_train.load_data(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = Datasets()
    dataset_val.load_data(args.dataset, "val")
    dataset_val.prepare()

    # Data augmentation
    augmentation = imgaug.augmenters.Sometimes(0.5, imgaug.augmenters.SomeOf(2, [imgaug.augmenters.Fliplr(0.5),
                                                                         imgaug.augmenters.Flipud(0.5),
                                                                         imgaug.augmenters.Affine(rotate=90),
                                                                         imgaug.augmenters.Superpixels(p_replace=0.5, n_segments=64),
                                                                         imgaug.augmenters.GaussianBlur(sigma=(0.0, 3.0))]))        
                                                                         #imgaug.augmenters.Affine(rotate=45)]))
        
    #MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes", "Fliplr", "Flipud", "CropAndPad", "Affine", "PiecewiseAffine"]

    
    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE,
                epochs=100, 
                layers="heads",
                augmentation=augmentation)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect bolts.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train'")
    parser.add_argument('--dataset', required=False,
                        default='./../../dataset/Corrosion_Condition_State_Classification/processed_512/',
                        metavar="/path/to/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=False,
                        default='coco',
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = Configuration()
    else:
        class InferenceConfig(Configuration):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train'".format(args.command))
