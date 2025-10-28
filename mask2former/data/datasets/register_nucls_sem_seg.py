import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg


NuCLS_SEM_SEG_CATEGORIES = [
   'tumor_any', 'nonTIL_stromal', 'sTIL', 'other_nucleus'
]

def register_nucls_sem_seg(root):
    """
    Register the NuCLS semantic segmentation dataset to Detectron2's DatasetCatalog.

    Args:
        root (str): The root directory where the NuCLS dataset is stored.
    """
    root = os.path.join(root, "NuCLS")
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "rgb", dirname)
        sem_seg_dir = os.path.join(root, "processed_mask", dirname)
        dataset_name = f"nucls_sem_seg_{name}"
        DatasetCatalog.register(
            dataset_name, lambda x=image_dir, y=sem_seg_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png")
        )
        MetadataCatalog.get(dataset_name).set(
            stuff_classes=NuCLS_SEM_SEG_CATEGORIES[:],
            image_root=image_dir,
            sem_seg_root=sem_seg_dir,
            evaluator_type="sem_seg",
            ignore_label= 255,
        )
        
_root = os.getenv("DETECTRON2_DATASETS", ".")
register_nucls_sem_seg(_root)