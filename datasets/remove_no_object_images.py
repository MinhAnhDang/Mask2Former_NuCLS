import numpy as np
import os
from PIL import Image
from collections import OrderedDict
import cv2

task = 'validation' # 'training' or 'validation'
data_dir = 'NuCLS'
training_image_dir = f'{data_dir}/rgb/{task}'
training_gt_mask_dir = f'{data_dir}/processed_mask/{task}'

image_files = os.listdir(training_image_dir)
mask_files = os.listdir(training_gt_mask_dir)
print(f'Number of images: {len(image_files)}')
print(f'Number of masks: {len(mask_files)}')

num_remove = 0
for file in image_files:
    img_path = os.path.join(training_image_dir, file)
    img = Image.open(img_path)
    img_array = np.array(img)
    
    mask_path = os.path.join(training_gt_mask_dir, file)
    mask = Image.open(mask_path)
    mask_array = np.array(mask)
    classes = np.unique(mask_array)
    classes = classes[classes != 255]
    if len(classes) == 0:
        #remove this file from folder
        os.remove(img_path)
        os.remove(mask_path)
        num_remove += 1
        print(f'Unique values in mask:', np.unique(mask_array))
        print(f'Unique values in mask for {file}: {np.unique(classes)}')
        print("Image name:", file)
    # print(f'Unique values in mask:', np.unique(mask_array))
    # print(f'Unique values in mask for {file}: {np.unique(classes)}')
    #
    #Check to see if 98 in mask
    if 98 in classes:
        print(f"{file} contain class 98")
print("Done")    
print("Num of remove:", num_remove)
    