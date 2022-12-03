# finalProject
This repository is created for the final project of CS5783. It includes:

RGB.zip  -- its the samples used to train and test object detection models

coco_eval.py -- downloaded code from https://github.com/pytorch/vision.git
coco_utils.py -- downloaded code from https://github.com/pytorch/vision.git
engine.py -- downloaded code from https://github.com/pytorch/vision.git
transforms.py -- downloaded code from https://github.com/pytorch/vision.git
utils.py -- downloaded code from https://github.com/pytorch/vision.git

gdalRW.py -- read and write image with geospatial information
peanutDataset.py  -- create pytorch dataset from training and testing samples
resnetssd.py  -- extract features from resnet50 for SSD model (This code referred from torchvision SSD code (https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssd.py ) and NVIDIA model code(https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/ssd/model.py )) 
accuracymetrics.py -- Calculated True Positive, False Positive, and False Negative under different IOU from the detection result
peanut_OD_RGB_finalProject.ipynb --  train SSD and Faster RCNN model, evaluate these model

To run the code, open 'peanut_OD_RGB_finalProject.ipynb' in colab. All detailed requirements are commented in the code.
