import numpy as np
import cv2

'''filter out partial object'''
def filter_masks(mask):

    obj_ids = np.unique(mask)
    index = 0
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

#SAVE DATA TO JPEG
def save_raster_to_jpg(raster, maxival, jpg_name):
    
    image = raster[:,:,:3]
    image = (image*255/maxival).astype(int)
    cv2.imwrite(jpg_name, image)
    
    
def save_mask_to_jpg(mask_data, jpg_name):

    outdata = mask_data
    outdata[outdata >0] = 255  #SPECIAL FOR MASKS
    
    cv2.imwrite(jpg_name, outdata)    