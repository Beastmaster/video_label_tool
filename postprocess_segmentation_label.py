'''
Post process: segmentation label
'''

import cv2
import numpy as np
from numba import jit

@jit
def postprocess_segmentation_label(label,threahold = 100):
    '''
    label should be np.uint8 type
    Eliminate small parts
    '''
    kernel = np.ones((10,10),np.uint8)
    total_ele = label.shape[0]*label.shape[1]
    for i in range(1,np.amax(label)):
        labeli = np.uint8(label==i)
        if np.count_nonzero(labeli)<threahold:
            label[labeli]=0
        else:
            closing = cv2.morphologyEx(labeli, cv2.MORPH_CLOSE, kernel)
            label = np.where(closing == 1, i, label)
    return label


