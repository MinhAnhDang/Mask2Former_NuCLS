import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances

__all__ = ["NuCLSSemSegDatasetMapper"]

class NuCLSSemSegDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """
    
    @configurable
    def __init__(
        self,
        is_train: bool = True,
        *,
        augmentations,
        image_format,
        ignore_label,   
        size_divisibility,
        remove_bkg,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        # fmt: off
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.image_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.remove_bkg = remove_bkg
        
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")
        
        
    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        if is_train:
            # Build augmentations for training
            augs = [
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                )
            ]
            if cfg.INPUT.CROP.ENABLED:
                augs.append(
                    T.RandomCrop_CategoryAreaConstraint(
                        cfg.INPUT.CROP.TYPE,
                        cfg.INPUT.CROP.SIZE,
                        cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                        cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    )
                )
            if cfg.INPUT.COLOR_AUG_SSD:
                augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
            augs.append(T.RandomFlip())    
        else:
            augs = utils.build_augmentation(cfg, is_train)
            
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label
        remove_bkg = not cfg.MODEL.MASK_FORMER.TEST.MASKS_BG    
        
        ret = {
            'is_train': is_train,
            'augmentations': augs,
            'image_format': cfg.INPUT.FORMAT,
            'ignore_label': ignore_label,
            'size_divisibility': cfg.INPUT.SIZE_DIVISIBILITY,
            'remove_bkg': remove_bkg,
        }
        
        return ret
    
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        
        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        else:
            sem_seg_gt = None
        # Define the augmentation input 
        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        # Apply the augmentation:
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            
        # Pad image and segmentation label here
        if self.is_train and self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()
        
        image_shape = (image.shape[-2], image.shape[-1]) #h,w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image
        
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()
            
        if "annotations" in dataset_dict:
            raise ValueError("Semantic Segmentation dataset should not have instance annotations")
        
        #Calculate image_mean_intensity
        mean_intensity_image = torch.mean(image, dim=0, dtype=float).numpy()
        
        #Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            
            masks = []
            classes_intensity = []
            for c in classes:
                masks.append(sem_seg_gt == c)
                intensity = np.mean(mean_intensity_image[sem_seg_gt==c], dtype=float)
                classes_intensity.append(intensity)
            
            #background intensity
            background_intensity = np.mean(mean_intensity_image[sem_seg_gt==self.ignore_label], dtype=float)
                
            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
                instances.mean_intensity_images = torch.zeros((0, image.shape[-2], image.shape[-1]))
                instances.classes_intensity = torch.zeros((0,))
                instances.background_intensity = torch.zeros((0,))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                mean_intensity_images = torch.stack([torch.from_numpy(np.ascontiguousarray(mean_intensity_image.copy())) for _ in range(len(masks))])
                classes_intensity = torch.stack([torch.as_tensor(x)for x in classes_intensity])
                instances.gt_masks = masks.tensor
                instances.classes_intensity = classes_intensity
                instances.mean_intensity_images = mean_intensity_images
                instances.background_intensity = torch.stack([torch.as_tensor(background_intensity) for _ in range(len(masks))])
            dataset_dict["instances"] = instances
        return dataset_dict
