import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model
from tools import AverageMeter, knn_monitor, Logger, file_exist_check
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from linear_eval import main as linear_eval
from datetime import datetime
import sys
#import wandb
import pandas as pd
import cv2
import imageio
from copy import deepcopy

def save_images(imgs, labels, name, fps=2):
    print("WRITING SAMPLE IMAGES")
    path_name = os.path.join("..", "images_" + name)
    if os.path.exists(path_name):
        shutil.rmtree(path_name)
    os.makedirs(path_name)
    assert len(imgs) == len(labels)
    images = []
    for i_save in range(len(imgs)):
        sample = imgs[i_save].numpy()

        # Convert to 0-1, (W, H, C)
        sample -= sample.min()
        sample /= sample.max()
        sample = (sample.transpose((1, 2, 0)) * 255)
        sample = cv2.resize(sample, (8 * sample.shape[1], 8 * sample.shape[0]), interpolation = cv2.INTER_CUBIC)

        # Save image
        images.append(sample)
    imageio.mimwrite(os.path.join(path_name, "movie.gif"), images, fps=fps)


def main(device, args):
    cifar_args = deepcopy(args)
    cifar_args.eval.num_classes = 10
    cifar_args.dataset_kwargs['dataset'] = 'cifar10'

    if args.no_augmentation:
        print("NO AUGMENTATION IID", flush=True)
        train_loader = torch.utils.data.DataLoader(
            dataset=get_dataset(
                transform=get_aug(train=False, train_classifier=True, **args.aug_kwargs),
                train=True,
                **args.dataset_kwargs),
            shuffle=True,
            batch_size=args.train.batch_size,
            **args.dataloader_kwargs
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=get_dataset(
                transform=get_aug(train=True, **args.aug_kwargs),
                train=True,
                **args.dataset_kwargs),
            shuffle=True,
            batch_size=args.train.batch_size,
            **args.dataloader_kwargs
        )

    memory_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=True, **args.aug_kwargs),
            train=True,
            **args.dataset_kwargs),
        shuffle=True,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs),
            train=False,
            **args.dataset_kwargs),
        shuffle=False,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )

    cifar_dataset_kwargs = cifar_args.dataloader_kwargs
    cifar_memory_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=True, **args.aug_kwargs),
            train=True,
            **cifar_args.dataset_kwargs),
        shuffle=True,
        batch_size=args.train.batch_size,
        **cifar_args.dataloader_kwargs
    )

    cifar_test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs),
            train=False,
            **cifar_args.dataset_kwargs),
        shuffle=False,
        batch_size=args.train.batch_size,
        **cifar_args.dataloader_kwargs
    )

    # define model
    model = get_model(args.model).to(device)
    model = torch.nn.DataParallel(model)

    # if args.wandb:
    #     wandb.watch(model)

    # define optimizer
    optimizer = get_optimizer(
        args.train.optimizer.name, model,
        lr=args.train.base_lr*args.train.batch_size/256,
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay)

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.train.warmup_epochs, args.train.warmup_lr*args.train.batch_size/256,
        args.train.num_epochs, args.train.base_lr*args.train.batch_size/256, args.train.final_lr*args.train.batch_size/256,
        len(train_loader),
        constant_predictor_lr=True # see the end of section 4.2 predictor
    )

    logger = Logger(tensorboard=args.logger.tensorboard, matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
    accuracy = 0
    # Start training
    if args.train.knn_monitor:
        train_accuracy = knn_monitor(model.module.backbone, memory_loader, memory_loader, device, k=min(args.train.knn_k, len(memory_loader.dataset)), hide_progress=args.hide_progress)
        test_accuracy = knn_monitor(model.module.backbone, memory_loader, test_loader, device, k=min(args.train.knn_k, len(memory_loader.dataset), hide_progress=args.hide_progress))
        print("before training (train, test) accuracy", train_accuracy, test_accuracy)

    train_accuracy = 0.
    test_accuracy = 0.
    train_std = 0.
    test_std = 0.

    global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')
    for epoch in global_progress:
        model.train()

        batch_loss = 0.
        batch_updates = 0

        local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)
        for idx, data in enumerate(local_progress):
            assert len(data) in [3, 2]
            if len(data) == 3:
                images1, images2, labels = data
                if type(images1) == list and len(images1) == 2 and type(images2) == list and len(images2) == 2:
                    images1 = images1[0]
                    images2 = images2[1]
            else:  # len(data) == 2
                images1, images2 = data[0]
                labels = data[1]
            if args.save_sample:
                save_images(torch.cat((images1, images2), 3), labels, "iid")
                return

            model.zero_grad()
            data_dict = model.forward(images1.to(device, non_blocking=True), images2.to(device, non_blocking=True))
            loss = data_dict['loss'].mean() # ddp
            data_dict['loss'] = loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            data_dict.update({'lr':lr_scheduler.get_lr()})

            local_progress.set_postfix(data_dict)
            logger.update_scalers(data_dict)

            batch_loss += loss.item()
            batch_updates += 1

        assert args.train.knn_monitor or args.linear_monitor
        if args.train.knn_monitor and epoch % args.train.knn_interval == 0:
            train_accuracy, train_features = knn_monitor(model.module.backbone, memory_loader, memory_loader, device, k=min(args.train.knn_k, len(memory_loader.dataset)), hide_progress=args.hide_progress)
            test_accuracy, test_features = knn_monitor(model.module.backbone, memory_loader, test_loader, device, k=min(args.train.knn_k, len(memory_loader.dataset)), hide_progress=args.hide_progress)
#        if args.linear_monitor and epoch % args.train.knn_interval == 0:
#            train_accuracy, test_accuracy, train_features, test_features = linear_eval(args, train_loader=memory_loader, test_loader=test_loader, model=model.module.backbone)

#            cifar_train_accuracy, cifar_test_accuracy, cifar_train_features, cifar_test_features = linear_eval(cifar_args, train_loader=cifar_memory_loader, test_loader=cifar_test_loader, model=model.module.backbone)

#        epoch_dict = {"Epoch": epoch, "Train Accuracy": train_accuracy, "Test Accuracy": test_accuracy,  "Loss": batch_loss / batch_updates, "Train Feature Standard Deviation": torch.std(train_features, dim=0).mean().item(), "Test Feature Standard Deviation": torch.std(test_features, dim=0).mean().item()}
        epoch_dict = {"Epoch": epoch, "Loss": batch_loss / batch_updates}
        print(epoch_dict)
        #if args.wandb:
        #    wandb.log(epoch_dict)

        global_progress.set_postfix(epoch_dict)
        logger.update_scalers(epoch_dict)

        if (epoch+1)%10 == 0:
            # Save checkpoint
            model_path = os.path.join(args.ckpt_dir, f"{args.name}_{datetime.now().strftime('%m%d%H%M%S')}.pth") # datetime.now().strftime('%Y%m%d_%H%M%S')
            torch.save({
                'epoch': epoch+1,
                'state_dict':model.module.state_dict()
            }, model_path)
            print(f"Model saved to {model_path}")
            with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
                f.write(f'{model_path}')


    train_accuracy, test_accuracy, train_features, test_features = linear_eval(args, train_loader=memory_loader, test_loader=test_loader, model=model.module.backbone)
    epoch_dict = {"Train Accuracy": train_accuracy, "Test Accuracy": test_accuracy, "Train Feature Standard Deviation": torch.std(train_features, dim=0).mean().item(), "Test Feature Standard Deviation": torch.std(test_features, dim=0).mean().item()}
    print(epoch_dict)

    cifar_train_accuracy, cifar_test_accuracy, cifar_train_features, cifar_test_features = linear_eval(cifar_args, train_loader=cifar_memory_loader, test_loader=cifar_test_loader, model=model.module.backbone)
    epoch_dict = { "Cifar Train Accuracy": cifar_train_accuracy, "Cifar Test Accuracy": cifar_test_accuracy}
    print(epoch_dict)

    # Save checkpoint
    model_path = os.path.join(args.ckpt_dir, f"{args.name}_{datetime.now().strftime('%m%d%H%M%S')}.pth") # datetime.now().strftime('%Y%m%d_%H%M%S')
    torch.save({
        'epoch': epoch+1,
        'state_dict':model.module.state_dict()
    }, model_path)
    print(f"Model saved to {model_path}")
    with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
        f.write(f'{model_path}')


if __name__ == "__main__":
    args = get_args()

    #if args.wandb:
    #    wandb_config = pd.json_normalize(vars(args), sep='_')
    #    wandb_config = wandb_config.to_dict(orient='records')[0]
    #    wandb.init(project='simsiam', config=wandb_config)

    print("Using device", args.device)

    main(device=args.device, args=args)

    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')



    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')
