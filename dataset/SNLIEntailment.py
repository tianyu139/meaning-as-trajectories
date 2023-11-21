# Meaning Representations from Trajectories in Autoregressive Models
# https://github.com/tianyu139/meaning-as-trajectories
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from datasets import load_dataset
from . import CompareTask
import numpy as np

class SNLIEntailment(CompareTask):
    def __init__(self, is_train=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng(139)
        self.dataset = load_dataset("snli")['train'] if is_train else load_dataset("snli")['validation']
        self.preprocess_dataset()
        
    def preprocess_example(self, example):
        # Filter examples with only 3 labels
        x = example['premise']
        y = example['hypothesis']
        label = example['label']

        if label == 0:
            if self.rng.random() < 0.5:
                return x, y, 0
            else:
                return y, x, 1
        else:
            return None, None, None