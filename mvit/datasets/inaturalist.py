import json
import os
import random
import re

from PIL import Image

import mvit.utils.logging as logging

import torch
from torchvision import transforms as transforms_tv
from torchvision import datasets
from torch.utils import data

from .build import DATASET_REGISTRY
from .transform import transforms_imagenet_train

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class INaturalist(data.Dataset):
    def __init__(self, cfg, mode, val_size=0.1, seed=42, **kwargs):
        self.cfg = cfg
        self.mode = mode
        self.data_path = cfg.DATA.PATH_TO_DATA_DIR
        assert mode in [
            'train',
            'val',
            'test',
        ], 'Split \'{}\' not supported for INaturalist'.format(mode)
        logger.info('Constructing INaturalist {}...'.format(mode))
        g = torch.Generator().manual_seed(seed)
        self.ds = datasets.INaturalist(
            root=self.data_path,
            version='2021_valid' if self.mode == 'test' else '2021_train',
            target_type='full',
            download=False,
        )
        if mode in {'train', 'val'}:
            val_len = round(len(self.ds) * val_size)
            train_len = len(self.ds) - val_len
            ds_train, ds_val = data.random_split(self.ds, [train_len, val_len],
                                                 generator=g)
            if mode == 'train':
                self.ds = ds_train
            else:
                self.ds = ds_val
        if cfg.DATA.PATH_TO_PRELOAD_IMDB != '':
            logger.info(f'Option DATA.PATH_TO_PRELOAD_IMDB='
                        f'{cfg.DATA.PATH_TO_PRELOAD_IMDB} invalid for '
                        f'INaturalist and will be ignored.')

    # def _prepare_im(self, im_path):
    def _prepare_im(self, im):
        # with pathmgr.open(im_path, 'rb') as f:
        #     with Image.open(f) as im:
        #         im = im.convert('RGB')
        # Convert HWC/BGR/int to HWC/RGB/float format for applying transforms
        train_size, test_size = (
            self.cfg.DATA.TRAIN_CROP_SIZE,
            self.cfg.DATA.TEST_CROP_SIZE,
        )

        if self.mode == 'train':
            aug_transform = transforms_imagenet_train(
                img_size=(train_size, train_size),
                color_jitter=self.cfg.AUG.COLOR_JITTER,
                auto_augment=self.cfg.AUG.AA_TYPE,
                interpolation=self.cfg.AUG.INTERPOLATION,
                re_prob=self.cfg.AUG.RE_PROB,
                re_mode=self.cfg.AUG.RE_MODE,
                re_count=self.cfg.AUG.RE_COUNT,
                mean=self.cfg.DATA.MEAN,
                std=self.cfg.DATA.STD,
            )
        else:
            t = []
            if self.cfg.DATA.VAL_CROP_RATIO == 0.0:
                t.append(
                    transforms_tv.Resize((test_size, test_size), interpolation=3),
                )
            else:
                # size = int((256 / 224) * test_size) # = 1/0.875 * test_size
                size = int((1.0 / self.cfg.DATA.VAL_CROP_RATIO) * test_size)
                t.append(
                    transforms_tv.Resize(
                        size, interpolation=3
                    ),  # to maintain same ratio w.r.t. 224 images
                )
                t.append(transforms_tv.CenterCrop(test_size))
            t.append(transforms_tv.ToTensor())
            t.append(transforms_tv.Normalize(self.cfg.DATA.MEAN, self.cfg.DATA.STD))
            aug_transform = transforms_tv.Compose(t)
        im = aug_transform(im)
        return im

    def __load__(self, index):
        # Load the image
        im_raw, label = self.ds[index]
        # Prepare the image for training / testing
        if self.mode == 'train' and self.cfg.AUG.NUM_SAMPLE > 1:
            im = []
            for _ in range(self.cfg.AUG.NUM_SAMPLE):
                crop = self._prepare_im(im_raw)
                im.append(crop)
            return im, label
        else:
            im = self._prepare_im(im_raw)
            return im, label

    def __getitem__(self, index):
        im, label = self.__load__(index)
        if isinstance(im, list):
            label = [label for _ in range(len(im))]
        return im, label

    def __len__(self):
        return len(self.ds)
