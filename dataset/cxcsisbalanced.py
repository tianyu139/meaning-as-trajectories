# Meaning Representations from Trajectories in Autoregressive Models
# https://github.com/tianyu139/meaning-as-trajectories
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import csv
import os


SIS_VAL_CSV_PATH = "./data/cxc/sis_val.csv"
COCO_VAL_PATH = "./data/coco/val2014"

class CXCSISBalanced(Dataset):
    def __init__(self, is_train=False, max_data_num=None, n_bins=5, return_paths=False):
        super().__init__()
        assert not is_train
        self.examples = []
        self.return_paths = return_paths
        self.max_data_num = max_data_num
        if self.max_data_num:
            assert self.max_data_num > 0

        if not is_train:
            reader = csv.reader(open(SIS_VAL_CSV_PATH, "r"), delimiter=',')
            next(reader)
            for row in reader:
                image1, image2, score, rating_type = row
                image1 = os.path.join(COCO_VAL_PATH, image1)
                image2 = os.path.join(COCO_VAL_PATH, image2)
                self.examples.append((image1, image2, float(score)))

        bins = np.arange(0, 5.0 + 5.0 / n_bins / 2, 5.0 / n_bins)
        bin_index = {}

        for i in range(n_bins):
            bin_start = bins[i]
            bin_end = bins[i + 1]

            bin_index[i] = [i for i in range(len(self.examples)) if
                            self.examples[i][2] >= bin_start and self.examples[i][2] < bin_end]

        smallest_bin_size = min([len(v) for v in bin_index.values()])
        truncated_idxs = []
        truncated_bins = []
        # Truncate everything to smallest bin
        for i in bin_index.keys():
            bin_index[i] = bin_index[i][:smallest_bin_size]
            truncated_idxs += bin_index[i]
            truncated_bins += [bins[i]] * smallest_bin_size

        self.examples = [self.examples[idx] for idx in truncated_idxs]

        rng = np.random.default_rng(139)
        # Random order
        rand_idxs = list(range(len(self.examples)))
        rng.shuffle(rand_idxs)

        if self.max_data_num is not None and self.max_data_num >= 0 and self.max_data_num < len(self.examples): # truncate datasets
            tmp_idxs = list(range(len(self.examples)))
            rng.shuffle(tmp_idxs)
            tmp_idxs = tmp_idxs[:self.max_data_num]
            print(f"Reducing dataset size from {len(self.examples)} to {self.max_data_num}...")
            self.examples = [self.examples[idx] for idx in tmp_idxs]

    def __getitem__(self, index):
        img_path_1, img_path_2, score = self.examples[index]
        if self.return_paths:
            return img_path_1, img_path_2, score
        else:
            return Image.open(img_path_1).convert('RGB'), Image.open(img_path_2).convert('RGB'), score

    def __iter__(self):
        for img_path_1, img_path_2, score in self.examples:
            if self.return_paths:
                yield img_path_1, img_path_2, score
            else:
                yield Image.open(img_path_1).convert('RGB'), Image.open(img_path_2).convert('RGB'), score

    def __len__(self):
        return len(self.examples)
