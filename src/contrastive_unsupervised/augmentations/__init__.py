from .simsiam_aug import SimSiamTransform
from .eval_aug import Transform_single
from .byol_aug import BYOL_transform
from .simclr_aug import SimCLRTransform
def get_aug(name='simsiam', image_size=224, train=True, train_classifier=None, double_images=False, single_aug=None):

    if train==True:
        if name == 'simsiam':
            augmentation = SimSiamTransform(image_size, double_images, single_aug)
        elif name == 'byol':
            augmentation = BYOL_transform(image_size, single_aug=single_aug)
        elif name == 'simclr':
            augmentation = SimCLRTransform(image_size)
        elif name == 'barlow':
            augmentation = BYOL_transform(image_size, single_aug=single_aug)
        else:
            raise NotImplementedError
    elif train==False:
        if train_classifier is None:
            raise Exception
        augmentation = Transform_single(image_size, train=train_classifier)
    else:
        raise Exception
    
    return augmentation








