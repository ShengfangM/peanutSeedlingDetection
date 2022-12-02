#!/usr/bin/env python
# coding: utf-8

# In[17]:


import os
import numpy as np

from osgeo import gdal, gdalnumeric
from gdalRW import getMeta, getAllData

import torchvision.transforms as T
from PIL import Image
import torch, utils


# In[26]:


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


# In[19]:


class peanutDroneDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, target_transforms):
        self.root = root
        self.transforms = transforms
        self.target_transforms = target_transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "Masks"))))
        
    def __getitem__(self,idx):
        # load images and masks
        
        img_path = os.path.join(self.root, "Images", self.imgs[idx])
        mask_path = os.path.join(self.root, "Masks", self.masks[idx])
        #print(img_path)
        #print(mask_path)
        
        #img = getAllData(img_path)
        img = Image.open(img_path).convert("RGB")

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        #mask = Image.open(mask_path)
        mask = getAllData(mask_path)
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
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            #pos = np.where(mask==i)
            #print(pos)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
                    
            boxes.append([xmin, ymin, xmax, ymax])
            #print([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        #masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        #target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            #img, target = self.transforms((img, target))
            img = self.transforms(img)
        if self.target_transforms is not None:    
            target = self.target_transforms(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


# In[20]:



def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    #if train:
        #transforms.append(T.RandomHorizontalFlip(0.5))
        #transforms.append(T.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,1.5),hue=(-0.1,0.1)))
    return T.Compose(transforms)


# In[24]:


# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle

# #test if data is correctly loaded
# #root = 'C:/Ma/Study/Peanut_Recognition/Images/resize_512/training'
# root = 'D:/Ma/Peanut_Recognition/data/Training/RGB/'
# #root = 'C:/Ma/PennFudanPed/PennFudanPed/'

# dataset = peanutDroneDataset(root, get_transform(train=True),target_transforms = None)

# data_loader = torch.utils.data.DataLoader(
#     dataset, batch_size=6, shuffle=True, num_workers=0,
#     collate_fn=utils.collate_fn)

# img,target = next(iter(data_loader))


# for l in range(6):
#     boxes = target[l]["boxes"] 
#     labels = target[l]["labels"] 
#     #masks = target[l]["masks"] 

#     imgi = img[l].permute(1,2,0) 

#     fig = plt.gcf()
#     fig.set_size_inches(18.5,10.5)

#     plt.imshow(imgi)

#     ax = plt.gca()

#     for i in range(len(boxes)):

#         ax.add_patch(Rectangle((boxes[i][0],boxes[i][1]),
#                                (boxes[i][2] - boxes[i][0]),
#                                (boxes[i][3] - boxes[i][1]),
#                               fill = False,
#                               edgecolor = 'red',
#                               linewidth=3))
#         ax.text(boxes[i][0],boxes[i][1], 'peanut')

#     fig = plt.gcf
#     plt.tick_params(labelbottom='on')
#     plt.show()

