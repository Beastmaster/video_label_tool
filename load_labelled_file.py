'''


'''

import glob
import os



def walk_image_dir(path,*sufix):
    '''
    sufix = '*.jpg','*.png'
    '''
    files = []
    for ss in sufix:
        files = files+glob.glob(os.path.join(path,ss))
    return files

def match_segmetation(image_names, label_names):
    pass

def set_output_dir(path):
    pass



if __name__ == "__main__":
    path = 'E:/DATA/Training_3.14/1/labelImage'
    walk_image_dir(path,'*.jpg',"*.png")


