#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import numpy as np
import cv2
from osgeo import gdal, gdalnumeric
from gdalRW import getMeta, getAllData
from PIL import ImageDraw, Image
import json

def filtermasks(mask):
    # instances are encoded as different colors
    obj_ids = np.unique(mask)
    #print(obj_ids)
    index = 0
    # reset all mask ids
    for i in range(len(obj_ids)):

        pos = np.where(mask == obj_ids[i])

        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        if ymax-ymin < 5 or xmax - xmin < 5 or (ymax-ymin)/(xmax - xmin )>1.5 or (ymax-ymin)/(xmax - xmin )<0.45:
        #if ymax-ymin < 4 or xmax - xmin < 4:
            mask[mask == obj_ids[i]] = 0 
        else:
            mask[mask == obj_ids[i]] = index
            index += 1      

    return mask


# In[4]:


#

class dataset2coco(object):
    def __init__(self, data_path, save_json_path="./coco.json", labels=['peanut']):
        """
        :param labelme_json: the list of all labelme json file paths
        :param save_json_path: the path to save new json
        """
                # ensure that they are aligned
        self.root = data_path
        self.imgs = list(sorted(os.listdir(os.path.join(self.root , "Images"))))
        self.masks = list(sorted(os.listdir(os.path.join(self.root , "Masks"))))
        
        
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        self.label = labels
        self.annID = 1
        self.height = 0
        self.width = 0

        self.save_json()

    def data_transfer(self):
        for num, imgfile in enumerate(self.imgs):

            self.images.append(self.image(os.path.join(self.root , "Images"), imgfile, num))
            self.annotation_set(os.path.join(self.root , "Masks"), self.masks[num], 'peanut', num)

        # Sort all text labels so they are in the same order across data splits.
        self.label.sort()
        for label in self.label:
            self.categories.append(self.category(label))
        for annotation in self.annotations:
            annotation["category_id"] = self.getcatid(annotation["category_id"])

    def image(self, path, file, num):
        image = {}
        img = cv2.imread(os.path.join(path,file))
        height, width = img.shape[:2]

        img = None
        image["height"] = height
        image["width"] = width
        image["id"] = num
        image["file_name"] = os.path.join("Images", file)

        self.height = height
        self.width = width

        return image

    def category(self, label):
        category = {}
        category["supercategory"] = label[0]
        category["id"] = len(self.categories)
        category["name"] = label[0]
        return category
    
    def annotation_set(self, path, mask, label, num):

        mask = getAllData(os.path.join(path,mask))
        #filter ... 
        mask = filtermasks(mask)
       
        obj_ids = np.unique(mask)
        #print(obj_ids)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]        
        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]  

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)

        for i in range(num_objs):
            points = np.where(masks[i])  
            self.annotations.append(self.annotation(points, label, num))
            self.annID += 1

    def annotation(self, points, label, num):
        annotation = {}
#         contour = np.array(points)
        x = points[1]
        y = points[0]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        poly1 = [(px + 0.5, py + 0.5) for px, py in zip(x, y)]
        annotation["segmentation"] = [list(np.asarray(poly1).flatten())]
        annotation["iscrowd"] = 0
        annotation["area"] = area
        annotation["image_id"] = num

        annotation["bbox"] = list(map(float, self.points2box(points)))

        annotation["category_id"] = label[0]  # self.getcatid(label)
        annotation["id"] = self.annID
        return annotation

    def getcatid(self, label):
        for category in self.categories:
            if label == category["name"]:
                return category["id"]
        print("label: {} not in categories: {}.".format(label, self.categories))
        exit()
        return -1


    def points2box(self, points):

        xmin = np.min(points[1])
        xmax = np.max(points[1])
        ymin = np.min(points[0])
        ymax = np.max(points[0])
            
        minpx = 0
        minpy = 0
        maxpx = self.width-1
        maxpy = self.height-1
        
        if xmin > 0:
            minpx = xmin -1 
            #px = np.append(px,minpx, axis=0)
        if xmax < self.width-1:
            maxpx = xmax+1
            #px = np.append(px,maxpx)
        if ymin > 0:
            minpy = ymin-1 
            #py = np.append(py,minpy, axis=0)
        if ymax < self.height-1:
            maxpy = ymax+1
            
        return [
            minpx,
            minpy,
            maxpx - minpx,
            maxpy - minpy,
        ]


    def data2coco(self):
        data_coco = {}
        data_coco["images"] = self.images
        data_coco["categories"] = self.categories
        data_coco["annotations"] = self.annotations
        return data_coco

    def save_json(self):
        print("save coco json")
        self.data_transfer()
        self.data_coco = self.data2coco()

        print(self.save_json_path)
        os.makedirs(
            os.path.dirname(os.path.abspath(self.save_json_path)), exist_ok=True
        )
        json.dump(self.data_coco, open(self.save_json_path, "w"), indent=4)


# In[9]:


# root = 'D:/Ma/Peanut_Recognition/data/Training/RGB_WHOLE/train/'

# aa = dataset2coco(root, "D:/Ma/Peanut_Recognition/data/Training/test_2/RGB/annotations.json", ['peanut'])

