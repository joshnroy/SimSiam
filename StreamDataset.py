import os
import os.path
import random
import sys
from tqdm import trange
from copy import deepcopy
from multiprocessing import Pool

import json
from PIL import Image
import numpy as np
import torch.utils.data as data
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as transforms

def instance_ordering(data_list, seed):
    # organize data by video
    total_videos = 0
    new_data_list = []
    temp_video = []
    for x in data_list:
        if x[3] == 0:
            new_data_list.append(temp_video)
            total_videos += 1
            temp_video = [x]
        else:
            temp_video.append(x)
    new_data_list.append(temp_video)
    new_data_list = new_data_list[1:]
    # shuffle videos
    random.seed(seed)
    random.shuffle(new_data_list)
    # reorganize by clip
    data_list = []
    for v in new_data_list:
        for x in v:
            data_list.append(x)
    return data_list # , new_data_list


def class_ordering(data_list, class_type, seed):
    # organize by class
    new_data_list = []
    # class_vids = []
    # class_vids_len = []
    # hist_data = {}
    for class_id in range(data_list[-1][0] + 1):
        class_data_list = [x for x in data_list if x[0] == class_id]
        if class_type == 'class_iid':
            # shuffle all class data
            random.seed(seed)
            random.shuffle(class_data_list)
        else:
            # shuffle clips within class
            class_data_list = instance_ordering(class_data_list, seed)
        new_data_list.append(class_data_list)
    # shuffle classes
    random.seed(seed)
    random.shuffle(new_data_list)
    # reorganize by class
    data_list = []
    for v in new_data_list:
        for x in v:
            data_list.append(x)
    return data_list


def make_dataset(data_list, ordering='class_instance', seed=666):
    """
    data_list
    for train: [class_id, clip_num, video_num, frame_num, bbox, file_loc]
    for test: [class_id, bbox, file_loc]
    """
    if not ordering or len(data_list[0]) == 3:  # cannot order the test set
        return data_list
    if ordering not in ['iid', 'class_iid', 'instance', 'class_instance']:
        raise ValueError('dataset ordering must be one of: "iid", "class_iid", "instance", or "class_instance"')
    if ordering == 'iid':
        # shuffle all data
        random.seed(seed)
        random.shuffle(data_list)
        return data_list
    elif ordering == 'instance':
        return instance_ordering(data_list, seed)
    elif 'class' in ordering:
        return class_ordering(data_list, ordering, seed)


class StreamDataset(data.Dataset):
    """Stream-51 Dataset Object

    Args:
        root (string): Root directory path of dataset.
        train (bool): load either training set (True) or test set (False) (default: True)
        ordering (string): desired ordering for training dataset: 'instance', 
            'class_instance', 'iid', or 'class_iid' (ignored for test dataset)
            (default: None)
        transform: A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform: A function/transform that takes
            in the target and transforms it.
        bbox_crop: crop images to object bounding box (default: True)
        ratio: padding for bbox crop (default: 1.10)
        seed: random seed for shuffling classes or instances (default=10)
     Attributes:
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, train=True, ordering=None, transform=None, target_transform=None, bbox_crop=True,
                 ratio=1.10, seed=10, small_dataset=False, temporal_jitter_range=0, preload=False):

        self.train = train
        self.preload = preload

        if self.train:
            data_list = json.load(open(os.path.join(root,'Stream-51_train.json')))
        else:
            data_list = json.load(open(os.path.join(root,'Stream-51_test.json')))
            data_list = [x for x in data_list if x[0] != 51]

        samples = make_dataset(data_list, ordering, seed=seed)
        if small_dataset:
            samples = samples[:1000]

        self.root = root
        self.loader = default_loader

        self.samples = samples
        self.targets = [s[0] for s in samples]


        self.transform = transform
        self.target_transform = target_transform

        self.bbox_crop = bbox_crop
        self.ratio = ratio

        self.classes = ["class_{:d}".format(i) for i in range(51)]
        self.temporal_jitter_range = temporal_jitter_range

        if self.preload:
            print("LOADING DATASET", flush=True)
            if False:
                self.loaded_images = {}
                for index in trange(len(self.samples)):
                    sample, target = self.get_item(index)
                    self.loaded_images[index] = (sample, target)
            else:
                with Pool(12) as p:
                    self.loaded_images = p.map(self.get_item, np.arange(len(self)))
            print("DONE LOADING", flush=True)


    def __getitem__(self, index):
        if self.preload:
            original_img, label = self.loaded_images[index]
        else:
            original_img, label = self.get_item(index)

        if self.transform is not None:
            new_img = self.transform(original_img)
            original_img = self.transform(original_img)
        else:
            new_img = original_img
        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.temporal_jitter_range == 0 and self.train:
            return original_img, new_img, label
        elif self.train:
            new_label = None
            new_index = np.minimum(index + self.temporal_jitter_range, len(self)-1) + 1
            while new_label != label:
                if False: # Use stochastic time jittering
                    new_index = np.minimum(np.maximum(index + np.random.randint(-self.temporal_jitter_range, self.temporal_jitter_range), 0), len(self)-1)
                else:
                    new_index -= 1
                if self.preload:
                    new_img, new_label = self.loaded_images[new_index]
                else:
                    new_img, new_label = self.get_item(new_index)

            if self.transform is not None:
                new_img = self.transform(new_img)

            return original_img, new_img, label
        else:
            return original_img, label

    def get_item(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        fpath, target = self.samples[index][-1], self.targets[index]
        sample = self.loader(os.path.join(self.root, fpath))
        if self.bbox_crop:
            bbox = self.samples[index][-2]
            cw = bbox[0] - bbox[1];
            ch = bbox[2] - bbox[3];
            center = [int(bbox[1] + cw / 2), int(bbox[3] + ch / 2)]
            bbox = [min([int(center[0] + (cw * self.ratio / 2)), sample.size[0]]),
                    max([int(center[0] - (cw * self.ratio / 2)), 0]),
                    min([int(center[1] + (ch * self.ratio / 2)), sample.size[1]]),
                    max([int(center[1] - (ch * self.ratio / 2)), 0])]
            sample = sample.crop((bbox[1],
                                  bbox[3],
                                  bbox[0],
                                  bbox[2]))

        resize_num = 128
        new_size = (resize_num, int(float(resize_num)/sample.size[0] * sample.size[1]))
        sample = sample.resize(new_size)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)
