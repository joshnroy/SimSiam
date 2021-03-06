from numpy.lib.shape_base import array_split
import torch
import torchvision
import os
from tqdm import trange
from multiprocessing import Pool
import numpy as np

class UCFDataset(torchvision.datasets.UCF101):
    def __init__(self, root, train, transform, temporal_jitter_range=0):
        super(UCFDataset, self).__init__(root, os.path.join(root, "ucfTrainTestlist"), 1, step_between_clips=1, train=train, transform=None, num_workers=9)

        self.temporal_jitter_range = temporal_jitter_range
        self.img_transform = transform


    def get_item_help(self, index):
        img, _, label = super(UCFDataset, self).__getitem__(index)
        img = img[0].permute((2, 0, 1)) / 255.

        return img, label

    def __getitem__(self, index):
        img, label = self.get_item_help(index)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.train:
            new_label = None
            new_idx = min(index + self.temporal_jitter_range, len(self)-1) + 1
            while new_label != label:
                new_idx -= 1
                new_img, new_label = self.get_item_help(index)
            if self.img_transform is not None:
                new_img = self.img_transform(new_img)
            return img, new_img, label
        else:
            return img, label

class UCFImageDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, train, transform, temporal_jitter_range=0, small_dataset=False, preload_dataset=True):
        if train:
            root += '-train'
        else:
            root += '-test'

        print("Starting to use Pytorch ImageFolder loader", flush=True)
        super(UCFImageDataset, self).__init__(root, transform=None)
        print("Done using Pytorch ImageFolder loader", flush=True)
        self.img_transform = transform
        self.train = train
        self.temporal_jitter_range = temporal_jitter_range
        self.small_dataset = small_dataset

        self.cache = {}
        if preload_dataset:
            print("Starting loading into memory", flush=True)
            if True:
                if self.small_dataset:
                    idxs = np.arange(super(UCFImageDataset, self).__len__())
                    np.random.shuffle(idxs)
                    idxs = idxs[:len(self)]
                    for i, original_idx in zip(np.arange(len(self)), idxs):
                        self.cache[i] = self.get_item_helper(original_idx)
                else:
                    for i in trange(len(self)):
                        self.cache[i] = self.get_item_helper(i)
            else:
                with Pool(64) as p:
                    self.cache = p.map(self.get_item_helper, np.arange(len(self)))
            print("Done loading into memory", flush=True)

    def get_item_helper(self, index):
        if index in self.cache:
            img, label = self.cache[index]
        else:
            img, label = super(UCFImageDataset, self).__getitem__(index)
            if self.small_dataset:
                print(index, label, flush=True)

        return img, label

    def __len__(self) -> int:
        if self.small_dataset:
            return 1000
        else:
            return super(UCFImageDataset, self).__len__()
    
    def __getitem__(self, index):
        img, label = self.get_item_helper(index)

        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.train:
            new_label = None
            new_idx = min(index + self.temporal_jitter_range, len(self)-1) + 1
            while new_label != label:
                assert new_idx >= index
                new_idx -= 1
                if new_idx == index:
                    new_img, new_label = self.get_item_helper(index)
                    break
                new_img, new_label = self.get_item_helper(new_idx)
            if self.img_transform is not None:
                new_img = self.img_transform(new_img)
            return img, new_img, label
        else:
            return img, label
