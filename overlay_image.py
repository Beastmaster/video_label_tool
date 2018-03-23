

import numpy as np
import cv2
from numba import jit
from postprocess_segmentation_label import postprocess_segmentation_label


colors = [ [255,0,0], [0,255,0], [0,0,255], 
            [255,255,0], [0,255,255] ,[255,0, 255], 
            [255,255,255],[255,255,255],[255,255,255],
            [255,255,255],[255,255,255],[255,255,255] ]


@jit
def overlay_image(image,mask,alpha=0.8):
    #mask = postprocess_segmentation_label(mask)
    image = image.copy()
    for i in range(1,np.amax(mask)):  # 10 person at most
        color = colors[i]
        for c in range(3):
            image[:, :, c] = np.where(mask == i,
                                    image[:, :, c] *
                                    (1 - alpha) + alpha * color[c],
                                    image[:, :, c])
    return image


@jit
def mask_on_new_background(ori_img,mask,background):
    '''
    Apply segmentation mask on original images, and apply to new background
    '''
    if (background.shape[-1]>3):
        background = background[...,0:3]
    sz = (ori_img.shape[1],ori_img.shape[0])
    background = cv2.resize(background,sz)
    mask = mask>0
    mask = np.stack((mask,mask,mask),axis=-1)
    background = np.where(mask,ori_img,background)
    return background



if __name__ == "__main__":
    import cv2
    fimg = "E:/DATA/Training_3.14/labelImage/image0000000.jpg"
    fimg2 = "E:/DATA/Training_3.14/7.png"
    flab = "E:/DATA/Training_3.14/img_seg/image0000090_mask.jpg"
    
    img = cv2.imread(fimg,-1)
    img2 = cv2.imread(fimg2,-1)
    lab = cv2.imread(flab,-1)

    nimg = mask_on_new_background(img,lab,img2)
    #nimg = overlay_image(img,lab)
    cv2.imshow("image",nimg)
    cv2.waitKey(0)