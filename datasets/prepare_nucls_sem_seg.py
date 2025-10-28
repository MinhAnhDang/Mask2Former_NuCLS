import os
import numpy as np
from PIL import Image
from collections import OrderedDict
import cv2


raw_to_super_codemap = OrderedDict({1: 1,
 2: 2,
 3: 3,
 4: 3,
 5: 2,
 6: 1,
 7: 2,
 8: 4,
 9: 99,
 10: 4,
 11: 4,
 12: 4,
 99: 99})


def process_mask_to_match_image_size(mask, image_shape):
        """Process mask to match the transformed image size."""
        # print("Original mask shape:", mask.shape)
        if image_shape[1] > mask.shape[1]:
            pad = ((0,0),(0, image_shape[1] - mask.shape[1]))
            mask = np.pad(mask, pad, mode='constant', constant_values=99)
        elif image_shape[1] < mask.shape[1]:
            mask = mask[:, :image_shape[1]]
        if image_shape[0] > mask.shape[0]:
            pad = ((0, image_shape[0] - mask.shape[0]),(0, 0))
            mask = np.pad(mask, pad, mode='constant', constant_values=99)
        elif image_shape[0] < mask.shape[0]:
            mask = mask[:image_shape[0], :]
        return mask
    
task = 'training' # 'training' or 'validation'
data_dir = 'NuCLS'
training_image_dir = f'{data_dir}/rgb/{task}'
training_gt_mask_dir = f'{data_dir}/mask'
training_save_dir = f'{data_dir}/processed_mask/{task}'
os.makedirs(training_save_dir, exist_ok=True)

image_files = os.listdir(training_image_dir)
mask_files = os.listdir(training_gt_mask_dir)
print(f'Number of images: {len(image_files)}')
print(f'Number of masks: {len(mask_files)}')

# image = Image.open(os.path.join(training_image_dir, image_files[0]))
# mask = Image.open(os.path.join(training_save_dir, mask_files[0]))

#Process and save masks
num_processed = 0
for file in image_files:
    file = 'TCGA-GM-A2DH-DX1_id-5ea40adcddda5f83989951a2_left-55826_top-58087_bottom-58368_right-56099.png'
    img_path = os.path.join(training_image_dir, file)
    img = Image.open(img_path)
    img_array = np.array(img)
    
    mask_path = os.path.join(training_gt_mask_dir, file)
    mask = Image.open(mask_path)
    mask_array = np.array(mask)
    # print("Image shape:", img_array.shape)
    # print("Mask shape:", mask_array.shape)
    
    processed_mask = np.zeros_like(mask_array[:,:,0], dtype=np.uint8)
    print(f'Unique values in mask for {file}: {np.unique(mask_array[:,:,0])}')
    print("Processed mask shape:", processed_mask.shape)
    for raw_value, super_value in raw_to_super_codemap.items():
        processed_mask[mask_array[:,:,0] == raw_value] = super_value #objects classes
        processed_mask[mask_array[:,:,0] == 99] = 0 # change unlabel to 0
        processed_mask[mask_array[:,:,0] == 253] = 0 #background
    #Verify processed mask size matches image size
    if processed_mask.shape != img_array.shape[:2]:
        print(f'Size mismatch for {file}: image size {img_array.shape[:2]}, mask size {processed_mask.shape}')
        processed_mask = cv2.resize(processed_mask, (img_array.shape[1], img_array.shape[0]), interpolation = cv2.INTER_NEAREST)
        print(f'After processing, mask size: {processed_mask.shape}')
    print(f'Unique values in processed_mask for {file}: {np.unique(processed_mask)}')
    processed_mask = processed_mask -1 # 0 (ignore) become 255, others are shifted by 1. Mean all raw classes 0,99,253 is now become 255
    print(f'Unique values in processed_mask for {file} after shifted: {np.unique(processed_mask)}')

    # Save processed mask
    Image.fromarray(processed_mask).save(os.path.join(training_save_dir, file))
    num_processed += 1
    print(f'Processed {num_processed} / {len(image_files)} masks', end='\r')
print('\nProcessing complete.')
