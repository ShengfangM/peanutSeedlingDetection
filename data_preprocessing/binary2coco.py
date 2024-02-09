import os
import numpy as np
import cv2
from PIL import ImageDraw, Image

# from osgeo import gdal, gdalnumeric
from io_utils import getMeta, getAllData
import json
from data_process import filter_masks


''' module that convert a binary mask image to coco.json format'''
class binary2coco(object):
    def __init__(self, data_path=None, save_json_path="./coco.json", labels = None):
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
        img = cv2.imread(os.path.join(path,file))
        height, width = img.shape[:2]

        img = None
        image = {
            "height": height,
            "width": width,
            "id": num,
            "file_name": os.path.join("Images", file),
        }
        self.height = height
        self.width = width
        return image

    def category(self, label):
        return {
            "supercategory": label[0],
            "id": len(self.categories),
            "name": label[0],
        }
    
    def annotation_set(self, path, mask, label, num):

        mask = getAllData(os.path.join(path,mask))
        #filter ... 
        mask = filter_masks(mask)
        
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
#         contour = np.array(points)
        x = points[1]
        y = points[0]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        poly1 = [(px + 0.5, py + 0.5) for px, py in zip(x, y)]
        annotation = {
            "segmentation": [list(np.asarray(poly1).flatten())],
            "iscrowd": 0,
        }
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
        print(f"label: {label} not in categories: {self.categories}.")
        exit()
        return -1

    def getbbox(self, points):
        polygons = points
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def points2box(self, points):

        xmin = np.min(points[1])
        xmax = np.max(points[1])
        ymin = np.min(points[0])
        ymax = np.max(points[0])

        minpx = xmin-1 if xmin > 0 else 0
        minpy = ymin-1 if ymin > 0 else 0
        
        maxpx = min(xmax + 1, self.width - 1)
        maxpy = min(ymax+1, self.height-1)

        return [
            minpx,
            minpy,
            maxpx - minpx,
            maxpy - minpy,
        ]
        

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        return {
            "images": self.images,
            "categories": self.categories,
            "annotations": self.annotations,
        }

    def save_json(self):
        print("save coco json")
        self.data_transfer()
        self.data_coco = self.data2coco()

        print(self.save_json_path)
        os.makedirs(
            os.path.dirname(os.path.abspath(self.save_json_path)), exist_ok=True
        )
        json.dump(self.data_coco, open(self.save_json_path, "w"), indent=4)


