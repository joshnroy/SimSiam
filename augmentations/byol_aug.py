import torchvision
from torchvision import transforms as T
from PIL import Image, ImageOps
try:
    from torchvision.transforms import GaussianBlur
except ImportError:
    from .gaussian_blur import GaussianBlur
    torchvision.transforms.GaussianBlur = GaussianBlur

imagenet_norm = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]


class BYOL_transform:  # Table 6
    def __init__(self, image_size, normalize=imagenet_norm, single_aug=None):

        # Exclude CIFAR
        transform1_p_blur = 1. if image_size > 32 else 0.
        transform2_p_blur = 0.1 if image_size > 32 else 0.

        resize_aug = T.Resize((image_size, image_size))
        augs = [T.ToTensor(), T.Normalize(*normalize)]

        if single_aug is None:
            augs1 = [
                T.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(
                    3.0/4.0, 4.0/3.0), interpolation=Image.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply(
                    [T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur( kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=transform1_p_blur),
            ] + augs

            augs2 = [
                T.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(
                    3.0/4.0, 4.0/3.0), interpolation=Image.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply(
                    [T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur( kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=transform2_p_blur),
                T.RandomApply([Solarization()], p=0.2),
            ]
        elif single_aug == 'RandomResizedCrop':
            augs1 = [resize_aug + T.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=( 3.0/4.0, 4.0/3.0), interpolation=Image.BICUBIC)] + augs
            augs2 = [resize_aug + T.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=( 3.0/4.0, 4.0/3.0), interpolation=Image.BICUBIC)] + augs
        elif single_aug == 'RandomHorizontalFlip':
            augs1 = [resize_aug, T.RandomHorizontalFlip(p=0.5)] + augs
            augs2 = [resize_aug, T.RandomHorizontalFlip(p=0.5)] + augs
        elif single_aug == 'ColorJitter':
            augs1 = [resize_aug, T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8)] + augs
            augs2 = [resize_aug, T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8)] + augs
        elif single_aug == 'RandomGrayscale':
            augs1 = [resize_aug, T.RandomGrayscale(p=0.2)] + augs
            augs2 = [resize_aug, T.RandomGrayscale(p=0.2)] + augs
        elif single_aug == 'GaussianBlur':
            augs1 = [resize_aug, T.RandomApply([T.GaussianBlur( kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=transform1_p_blur)] + augs
            augs2 = [resize_aug, T.RandomApply([T.GaussianBlur( kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=transform2_p_blur)] + augs
        elif single_aug == 'Solarization':
            augs1 = [resize_aug] + augs
            augs2 = [resize_aug, T.RandomApply([Solarization()], p=0.2)] + augs
        elif single_aug == 'Nothing':
            augs1 = [resize_aug] + augs
            augs2 = [resize_aug] + augs
        else:
            raise ValueError("single_aug", single_aug, " must be a BYOL augmentation")

        self.transform1 = T.Compose(augs1)
        self.transform2 = T.Compose(augs2)

    def __call__(self, x):
        x1 = self.transform1(x)
        x2 = self.transform2(x)
        return x1, x2


class Transform_single:
    def __init__(self, image_size, train, normalize=imagenet_norm):
        # self.denormalize = Denormalize(*imagenet_norm)
        if train == True:
            self.transform = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(
                    3.0/4.0, 4.0/3.0), interpolation=Image.BICUBIC),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(*normalize)
            ])
        else:
            self.transform = T.Compose([
                T.Resize(int(image_size*(8/7)),
                                  interpolation=Image.BICUBIC),  # 224 -> 256
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(*normalize)
            ])

    def __call__(self, x):
        return self.transform(x)


class Solarization():
    # ImageFilter
    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, image):
        return ImageOps.solarize(image, self.threshold)
