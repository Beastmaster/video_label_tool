

import numpy as np
import cv2 




def apply_modify(image,begin,end,mode,feed_val = 1):
    '''
    image: input image
    rect: [x,y,x1,y1] block
    mode: "reserve", "delete"
    feed_val: default to be 1
    '''
    if begin is None:
        return image
    if end is None:
        return image
    if image is None:
        return image


    [x,y,x1,y1] = [min(begin[1],end[1]), 
                    min(begin[0],end[0]), 
                    max(begin[1],end[1]),
                    max(begin[0],end[0]) ]
    image = image.copy()
    mask = np.zeros_like(image,dtype=np.bool)
    if(len(image.shape)>2):
        mask[x:x1+1,y:y1+1,...] = True
    else:
        mask[x:x1+1,y:y1+1] = True
    if mode == "reserve":
        image = np.where(mask,image,0)
    else:
        image = np.where(mask,0,image)
    return image.astype(np.uint8)



def draw_points_on_label(label,points,mode="add",test = False):
    '''
    label: 1 channel label file (np.uint8)
    points: (x,y,size)
    mode: 'add'/'earse'
    '''
    # expand channel first
    if len(label.shape)<3 or label.shape[-1]==1:
        label = np.stack((label,label,label),-1)
    if mode=="add":
        color = (1,0,0)
    else:
        color = (0,0,0)
    for pt in points:
        label = cv2.circle(label,(pt[0],pt[1]),int(pt[2]/2),color,-1)
    if test:
        return label
    else:
        return label[...,0]
    



if __name__ == "__main__":
    fname = 'E:/DATA/Training_3.14/img_seg/image0000000_mask.jpg'
    img = cv2.imread(fname,-1)
    cv2.imshow("image",img*100)
    cv2.waitKey(0)




