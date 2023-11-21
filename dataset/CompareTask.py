# Meaning Representations from Trajectories in Autoregressive Models
# https://github.com/tianyu139/meaning-as-trajectories
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import string

import numpy as np
from torch.utils.data import Dataset


class CompareTask(Dataset):
    def __init__(self, max_data_num=None, end_with_fullstop=False):
        super().__init__()
        self.examples = []
        self.max_data_num = max_data_num
        self.end_with_fullstop = end_with_fullstop
        if self.max_data_num:
            assert self.max_data_num > 0

    def preprocess_example(self):
        raise NotImplementedError()

    def preprocess_dataset(self):
        for example in self.dataset:
            example = self.preprocess_example(example)
            if example[0] is None:
                continue
            self.examples.append(example)
 
        if self.max_data_num is not None and self.max_data_num >= 0 and self.max_data_num < len(self.examples): # truncate dataset
            tmp_idxs = list(range(len(self.examples)))
            rng = np.random.default_rng(139)
            rng.shuffle(tmp_idxs)
            tmp_idxs = tmp_idxs[:self.max_data_num]
            print(f"Reducing dataset size from {len(self.examples)} to {self.max_data_num}...")
            self.examples = [self.examples[idx] for idx in tmp_idxs]

        if self.end_with_fullstop:
            self.add_punctuation()

    def __len__(self):
        return len(self.examples)

    def add_punctuation_to_string(self, str):
        puncs = string.punctuation
        if str[-1] in puncs:
            return str
        else:
            return str + '.'

    def add_punctuation(self):
        for i in range(len(self.examples)):
            input_str, output_str, y = self.examples[i]
            input_str = self.add_punctuation_to_string(input_str)
            if type(output_str) == type([]):
                output_str = [self.add_punctuation_to_string(s) for s in output_str]
            elif type(output_str) == type(""):
                output_str = self.add_punctuation_to_string(output_str)
            else:
                raise ValueError("Unexpected type")
            self.examples[i] = (input_str, output_str, y)

    def __getitem__(self, index):
        input_str, output_str, y = self.examples[index]
        return input_str, output_str, y

    def __iter__(self):
        for input_str, output_str, y in self.examples:
            yield input_str, output_str, y
