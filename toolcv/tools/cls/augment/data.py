"""
pip install timm
pip install -U albumentations
pip install git+https://github.com/aleju/imgaug.git
"""
from torchvision import transforms as T
from PIL import Image
import torch
import numpy as np

try:
    from timm.data import RandAugment, rand_augment_ops, \
        AutoAugment, auto_augment_policy, Mixup, FastCollateMixup, \
        ToTensor, ToNumpy, RandomResizedCropAndInterpolation, create_transform, \
        IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
except:
    pass

if __name__ == "__main__":
    img = Image.open("000000041990.jpg")

    # transform = T.Compose([
    #     RandomResizedCropAndInterpolation((224, 224)),
    #     # RandAugment(rand_augment_ops()),
    #     # AutoAugment(auto_augment_policy()),
    #     # Mixup(),
    #     T.ToTensor(),
    #     T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    # ])

    transform = create_transform((224, 224), is_training=True)

    new_img = transform(img)

    new_img = (new_img.permute(1, 2, 0) * torch.tensor(IMAGENET_DEFAULT_STD)[None, None] + \
              torch.tensor(IMAGENET_DEFAULT_MEAN)[None, None])*255.

    new_img= Image.fromarray(new_img.clamp(0,255).cpu().numpy().astype(np.uint8))

    new_img.show()

