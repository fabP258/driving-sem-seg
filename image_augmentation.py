import albumentations as alb
from abc import ABC, abstractmethod
import cv2
import numpy as np


class ImageAugmentor(ABC):
    def __init__(self):
        self._augmentations: list = []

    @abstractmethod
    def add_transforms(self):
        pass

    def get_transform(self, height: int, width: int):
        self.add_transforms()
        self._augmentations.append(
            alb.PadIfNeeded(
                min_height=ImageAugmentor.ceil_to_multiple(height),
                min_width=ImageAugmentor.ceil_to_multiple(width),
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
            )
        )
        return alb.Compose(self._augmentations, p=1.0)

    @staticmethod
    def ceil_to_multiple(x: int, k: int = 32) -> int:
        ceiled_x = int(k * (np.ceil(x / k)))
        return ceiled_x


class NoImageAugmentor(ImageAugmentor):
    """Performs only padding to a integer multiple without any augmentations"""

    def __init__(self):
        super().__init__()

    def add_transforms(self):
        return


class LightImageAugmentations(ImageAugmentor):
    def __init__(self):
        super().__init__()

    def add_transforms(self):
        self._augmentations.append(alb.HorizontalFlip(p=0.5))
        self._augmentations.append(alb.GaussNoise(var_limit=(10.0, 20.0), p=0.2))
        self._augmentations.append(
            alb.OneOf(
                [
                    alb.CLAHE(p=1.0),
                    alb.RandomBrightnessContrast(p=1.0),
                    alb.RandomGamma(p=1.0),
                ],
                p=0.5,
            )
        )
        self._augmentations.append(
            alb.OneOf(
                [
                    alb.Sharpen(p=1.0),
                    alb.Blur(blur_limit=3, p=1.0),
                    alb.MotionBlur(blur_limit=3, p=1.0),
                ],
                p=0.5,
            )
        )
