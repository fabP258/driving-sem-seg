from torch.utils.data import Dataset
from typing import Union, Set
from pathlib import Path
from PIL import Image
import albumentations as alb
import torch
import numpy as np
from os import listdir
from image_augmentation import ImageAugmentor, NoImageAugmentor


class SegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        file_names: Set[str] = None,
        augmentor: ImageAugmentor = NoImageAugmentor(),
        image_size: tuple = (576, 448),
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        image_names = set(listdir(image_dir))
        mask_names = set(listdir(mask_dir))
        self.file_names = image_names & mask_names
        if file_names:
            self.file_names = self.file_names & file_names
        self.file_names = list(self.file_names)

        self.augmentor = augmentor

        # mapping from grayscale to class indices
        self.class_values = [42, 76, 90, 124, 161]
        self.value_to_index = {
            value: idx for idx, value in enumerate(self.class_values)
        }
        self.index_to_value = {
            idx: value for idx, value in enumerate(self.class_values)
        }

        self.image_size = image_size

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # load the image
        raw_image = Image.open(self.image_dir / self.file_names[idx])
        rgb_image = raw_image.convert("RGB")
        rgb_image = rgb_image.resize(self.image_size)
        image = np.array(rgb_image)

        # load the mask
        raw_mask = Image.open(self.mask_dir / self.file_names[idx])
        gs_mask = raw_mask.convert("L")
        gs_mask = gs_mask.resize(self.image_size, Image.NEAREST)
        mask_array = np.array(gs_mask)
        index_mask = np.vectorize(self.value_to_index.get)(mask_array)

        # image augmentation
        augmentation_transform = self.augmentor.get_transform(
            height=self.image_size[1], width=self.image_size[0]
        )
        augmented = augmentation_transform(image=image, mask=index_mask)
        image = augmented["image"]
        index_mask = augmented["mask"]

        # to tensor
        mask = torch.tensor(index_mask, dtype=torch.long)
        image = torch.tensor(np.transpose(image / 255, (2, 0, 1)), dtype=torch.float32)

        return image, mask
