#%%
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

#%%
args=get_args(inputs=["--config_file=configs/simsiam_stream51.yaml", "--data_dir=../stream_data", "--log_dir=../alignment_logs", "--ckpt_dir=.cache/jitter0", "--preload_dataset", "--bbox_crop", "--eval_from=alignment_models/stream51-cifar_time_jittering_deterministic0.pth"])

# %%
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
args.eval.num_epochs = 10

args.dataset_kwargs['ordering'] = 'instance'
train_loader = torch.utils.data.DataLoader(
    dataset=get_dataset( 
        transform=get_aug(train=True, train_classifier=False, **args.aug_kwargs), 
        train=True, 
        **args.dataset_kwargs
    ),
    batch_size=args.eval.batch_size,
    shuffle=True,
    **args.dataloader_kwargs
    )

test_loader = torch.utils.data.DataLoader(
    dataset=get_dataset(
        transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), 
        train=False,
        **args.dataset_kwargs
    ),
    batch_size=args.eval.batch_size,
    shuffle=True,
    **args.dataloader_kwargs
    )
# %%
if args.eval_from is not None:
    print("Loading model from", args.eval_from)
    model = get_backbone(args.model.backbone)
    save_dict = torch.load(args.eval_from, map_location='cpu')
    msg = model.load_state_dict({k[9:]:v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')}, strict=True)
    print(msg)

    model.to(args.device)
else:
    print("Specify an eval_from")

# %%
with torch.no_grad():
    all_train_features = []
    train_outputs = {}
    for x in tqdm(train_loader):
        images = x[0]
        labels = x[-1]

        feature = model(images.to(args.device, non_blocking=True))
        for i_label in range(len(labels)):
            l = labels[i_label].item()
            f = feature[i_label].detach().cpu()
            if l not in train_outputs:
                train_outputs[l] = []
            train_outputs[l].append(f)
            all_train_features.append(f)

    all_train_features = torch.stack(all_train_features)
    for l in train_outputs:
        train_outputs[l] = (torch.stack(train_outputs[l]) - all_train_features.mean()) / all_train_features.std()
    all_train_features = (all_train_features - all_train_features.mean()) / all_train_features.std()

# %%
with torch.no_grad():
    all_test_features = []
    test_outputs = {}
    for x in tqdm(test_loader):
        images = x[0]
        labels = x[-1]

        feature = model(images.to(args.device, non_blocking=True))
        for i_label in range(len(labels)):
            l = labels[i_label].item()
            f = feature[i_label].detach().cpu()
            if l not in test_outputs:
                test_outputs[l] = []
            test_outputs[l].append(f)
            all_test_features.append(f)

    all_test_features = torch.stack(all_test_features)
    for l in test_outputs:
        test_outputs[l] = (torch.stack(test_outputs[l]) - all_test_features.mean()) / all_test_features.std()
    all_test_features = (all_test_features - all_test_features.mean()) / all_test_features.std()
# %%

def align_loss(x, y, device, alpha=2, batch_size=512):
    align_loss = 0.
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size].to(device)
        y_batch = y[i:i+batch_size].to(device)
        align_loss += (x_batch - y_batch).norm(p=2, dim=1).pow(alpha).mean().item()
    return align_loss

def uniform_loss(x, device, t=2, batch_size=512):
    uniform_loss = 0.
    n = 0
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size].to(device)
        uniform_loss = (n / (n + 1.)) * uniform_loss + (1 / (n + 1.)) * torch.pdist(x_batch, p=2).pow(2).mul(-t).exp().mean().log().item()
        n += 1.
        print(uniform_loss)
    return uniform_loss
# %%

with torch.no_grad():
    train_uniform_metric = uniform_loss(all_train_features, args.device)
    test_uniform_metric = uniform_loss(all_test_features, args.device)
train_uniform_metric
test_uniform_metric
# %%
