import os
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from tqdm import tqdm, trange
from arguments import get_args
from augmentations import get_aug
from models import get_model, get_backbone
from tools import AverageMeter
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from sklearn.manifold import TSNE
import plotly.express as px
import io
from PIL import Image
import imageio
import numpy as np
import cv2
import random

def main(args, train_loader=None, test_loader=None, model=None, tsne_visualization=False):

    initially_none = train_loader is None

    if train_loader is None:
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        args.dataset_kwargs['ordering'] = 'instance'
        train_loader = torch.utils.data.DataLoader(
            dataset=get_dataset( 
                transform=get_aug(train=True, train_classifier=False, mean_std=[[0., 0., 0.], [1., 1., 1.]], **args.aug_kwargs), 
                train=True, 
                **args.dataset_kwargs
            ),
            batch_size=1,
            shuffle=True,
            **args.dataloader_kwargs
        )

    # Iterate through loader
    
    n = 0
    mean = torch.zeros(3)
    local_progress = tqdm(train_loader)
    for x in local_progress:
        image = x[0][0]
        n += np.prod(image.shape[-2:])
        image_mean = torch.sum(image, dim=[1, 2])
        mean += image_mean

    mean /= n
    print("mean", mean)

    var = torch.zeros(3)
    local_progress = tqdm(train_loader)
    for x in local_progress:
        image = x[0][0]
        for i in range(image.shape[0]):
            image[i] -= mean[i]
        image_var = image**2.
        image_var = torch.sum(image_var, dim=[1, 2])
        var += image_var

    var /= (n - 1)
    print("std", torch.sqrt(var))


    return mean, var





if __name__ == "__main__":
    main(args=get_args(), tsne_visualization=True)

