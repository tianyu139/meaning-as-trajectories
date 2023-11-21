# Meaning Representations from Trajectories in Autoregressive Models
# https://github.com/tianyu139/meaning-as-trajectories
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from torch.utils.data import Dataset
import os
import json
from PIL import Image
import numpy as np
import string


VAL_JSON_PATH = "./data/cxc/sits_val_combined.json"
COCO_PATH = "./data/coco/"

class CXCSITSBalanced(Dataset):
    def __init__(self, is_train=False, max_data_num=None, end_with_fullstop=False, n_bins=20, discretize=False,
                 coco_path=None, val_json_path=None):
        super().__init__()
        assert not is_train
        if coco_path is None:
            coco_path = COCO_PATH
        if val_json_path is None:
            val_json_path = VAL_JSON_PATH

        self.examples = []
        self.max_data_num = max_data_num
        self.end_with_fullstop = end_with_fullstop
        if self.max_data_num:
            assert self.max_data_num > 0

        if not is_train:
            with open(val_json_path) as f:
                dataset = json.load(f)

        with open(os.path.join(coco_path, 'dataset.json')) as f:
            og = json.load(f)

        sentence_ids = {}
        for x in og['images']:
            for s in x['sentences']:
                sid = s['sentid']
                text = s['raw']
                assert sid not in sentence_ids
                sentence_ids[sid] = text

        dataset = [x for x in dataset['images'] if 'cxc_scores' in x and len(x['cxc_scores']) > 0]

        image_text_score = []

        for d in dataset:
            img_path = os.path.join(coco_path, d['filepath'], d['filename'])
            for s in d['cxc_scores']:
                target, score, _ = s
                if type(target) == type(""):
                    continue
                else:
                    text = sentence_ids[target]

                if end_with_fullstop:
                    text = self.add_punctuation_to_string(text.strip())
                    text = text[0].upper() + text[1:]
                image_text_score.append((img_path, text, float(score)))

        self.examples = image_text_score

        bins = np.arange(0, 5.0 + 5.0/n_bins/2, 5.0/n_bins)
        bin_index = {}

        for i in range(n_bins):
            bin_start = bins[i]
            bin_end = bins[i+1]

            bin_index[i] = [i for i in range(len(self.examples)) if self.examples[i][2] >= bin_start and self.examples[i][2] < bin_end]

        smallest_bin_size = min([len(v) for v in bin_index.values()])
        truncated_idxs = []
        truncated_bins = []
        # Truncate everything to smallest bin
        for i in bin_index.keys():
            bin_index[i] = bin_index[i][:smallest_bin_size]
            truncated_idxs += bin_index[i]
            truncated_bins += [bins[i]] * smallest_bin_size

        self.examples = [self.examples[idx] for idx in truncated_idxs]

        if discretize:
            for i in range(len(self.examples)):
                img_path, text, score = self.examples[i]
                self.examples[i] = (img_path, text, truncated_bins[i])

        rng = np.random.default_rng(139)
        # Random order
        rand_idxs = list(range(len(self.examples)))
        rng.shuffle(rand_idxs)
        self.examples = [self.examples[idx] for idx in rand_idxs]

        if self.max_data_num is not None and self.max_data_num >= 0 and self.max_data_num < len(self.examples): # truncate dataset
            tmp_idxs = list(range(len(self.examples)))
            rng.shuffle(tmp_idxs)
            tmp_idxs = tmp_idxs[:self.max_data_num]
            print(f"Reducing dataset size from {len(self.examples)} to {self.max_data_num}...")
            self.examples = [self.examples[idx] for idx in tmp_idxs]

    def __getitem__(self, index):
        img_path, text, score = self.examples[index]
        return Image.open(img_path).convert('RGB'), text, score

    def __iter__(self):
        for img_path, text, score in self.examples:
            yield Image.open(img_path).convert('RGB'), text, score

    def __len__(self):
        return len(self.examples)

    def add_punctuation_to_string(self, str):
        puncs = string.punctuation
        if str[-1] in puncs:
            return str
        else:
            return str + '.'








