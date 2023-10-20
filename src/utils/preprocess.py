import os
import torch

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from typing import List

from PIL import Image


def read_dataset_images(dataset: Dataset, img_idx: List[int], normalize_transform):
    """Read images from PyTorch Dataset.
    
    Args:
        normalize_transform: the normalization transform. Model takes normalized inputs while visualising samples needs the original.
    
    Returns:
        img_list (List[Tensor]): unnormalized.
        batch (Tensor): normalized data as inputs to model.
        target_list (List[Tensor]): targets are also retrieved from dataset.
    
    """
    img_list = [dataset[idx][0] for idx in img_idx]
    target_list = [dataset[idx][1] for idx in img_idx]
    batch = torch.cat([normalize_transform(img).unsqueeze(0) for img in img_list], dim=0)

    return img_list, batch, target_list



def read_external_images(paths: str, normalize_transform):
    """Read images from external files or directory.
    
    All files in the directory will be read. Make sure the directory only includes the images.
    
    Args:
        paths (str): file or directory paths (allow multiple, split with space)
        normalize_transform: the normalization transform. Model takes normalized inputs while visualising samples needs the original.
        
    Returns:
        img_list (List[Tensor]): unnormalized.
        batch (Tensor): normalized data as inputs to model.
    
    """

    img_path_list = []
    path_list = paths.split(" ")
    for path in path_list:
        if os.path.isdir(path):
            img_path_list.extend(os.listdir(path))
        elif os.path.isfile(path):
            img_path_list.append(path)
    
    img_list = [ToTensor()(Image.open(img_path)) for img_path in img_path_list]
    
    batch = torch.cat([normalize_transform(img).unsqueeze(0) for img in img_list], dim=0)
    
    
    return img_list, batch

