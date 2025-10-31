import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#TODO Calculate average mask
def calculate_average_mask(src_masks,src_logits, num_masks_per_class):
    src_masks = src_masks.sigmoid()#BX, H,W
    _,H,W = src_masks.shape
    src_average_masks = torch.einsum('bhw,b->bhw', src_masks, src_logits)
    src_average_masks = src_average_masks.reshape(-1,num_masks_per_class, H, W)
    # print('src_average_masks', src_average_masks.shape)
    src_average_masks = src_average_masks.mean(dim=1)
    return src_average_masks
    
#TODO Calculate uncertain mask
def calculate_uncertain_mask(src_average_masks, theta, threshold_uc):
    entropy_mask = -0.5*(src_average_masks * torch.log(src_average_masks+theta) + (1-src_average_masks)*torch.log(1-src_average_masks+theta))
    flatten_entropy_mask = entropy_mask.flatten(start_dim=1)
    min_value = torch.min(flatten_entropy_mask, dim=1).values
    max_value = torch.max(flatten_entropy_mask, dim=1).values
    threshold = torch.as_tensor(min_value + threshold_uc*(max_value-min_value))
    uncertain_masks = src_average_masks > threshold[:,None,None]
    return uncertain_masks

#TODO Refine uncertain mask
def refine_uncertain_mask(src_average_masks:torch.Tensor, 
                          uncertain_masks, 
                          mean_intensity_images: torch.Tensor, 
                          class_intensity: torch.Tensor, #BX
                          back_ground_intensity:torch.Tensor,
                          low_bound,
                          high_bound):#1
    src_average_flatten = src_average_masks.flatten(start_dim=1)
    min_value = torch.min(src_average_flatten, dim=1).values
    max_value = torch.max(src_average_flatten, dim=1).values
    threshold = torch.as_tensor(min_value + 0.5*(max_value-min_value))
    binary_masks = torch.where(src_average_masks > threshold[:,None,None], 1, 0)
    binary_masks = binary_masks.unsqueeze(0)
    binary_masks = F.interpolate(binary_masks.float(),
                                 size=(mean_intensity_images.shape[-2], mean_intensity_images.shape[-1]),
                                 mode="bilinear",
                                 align_corners=False,).squeeze(0)
    uncertain_masks = F.interpolate(uncertain_masks.float().unsqueeze(0),
                                    size=(mean_intensity_images.shape[-2], mean_intensity_images.shape[-1]),
                                 mode="bilinear",
                                 align_corners=False,).squeeze(0)
    # ========== FN correction ==========
    inverse_binary_masks = 1 - binary_masks
    FN_UH = uncertain_masks * inverse_binary_masks #BX, H, W
    FN_UH = FN_UH*mean_intensity_images
    # print('Class_intensity', class_intensity.shape)
    # print('FN_UH', FN_UH.shape)
    FN_correction = torch.logical_and((class_intensity[:, None, None]*low_bound < FN_UH), (FN_UH > class_intensity[:, None, None]*high_bound))       
    # ========== FP correction ==========
    FP_UH = binary_masks * uncertain_masks
    FP_UH = FP_UH*mean_intensity_images
    FP_correction = torch.logical_and((back_ground_intensity[:, None, None]*low_bound< FP_UH), (FP_UH < back_ground_intensity[:, None, None]*high_bound) )
    #Final mask
    final_masks = torch.logical_and(binary_masks, FN_correction)
    final_masks = torch.logical_and(final_masks, torch.logical_not(FP_correction))
    
    return final_masks

#TODO Redefined GroundTruth Mask
@torch.no_grad()
def refine_targets(src_masks: torch.Tensor,
                   targets: torch.Tensor,
                   final_masks: torch.Tensor):
    pseudo_groundtruth = torch.logical_and(targets, final_masks)
    ignore_region = pseudo_groundtruth != targets
    #Modify src_masks in ingnore_region, edit logit to 10e-7 to ignore loss
    src_masks = F.interpolate(src_masks.float().unsqueeze(0),
                              size=(targets.shape[-2], targets.shape[-1]),
                              mode="bilinear",
                              align_corners=False,).squeeze(0)
    src_masks[ignore_region] = 10e-7
    
    return src_masks, pseudo_groundtruth.float()
    
    
    