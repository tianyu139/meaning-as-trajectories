# Meaning Representations from Trajectories in Autoregressive Models
# https://github.com/tianyu139/meaning-as-trajectories
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from datasets import load_dataset
from . import CompareTask

class STSB(CompareTask):
    def __init__(self, is_train=False, test_set=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_score_dataset = True
        self.dataset = load_dataset("mteb/stsbenchmark-sts")
        if not test_set:
            self.dataset = self.dataset['train'] if is_train else self.dataset['validation']
        else:
            assert is_train is False
            self.dataset = self.dataset['test']

        self.preprocess_dataset()
        
    def preprocess_example(self, example):
        # Filter examples with only 3 labels
        x1 = example['sentence1']
        x2 = example['sentence2']
        score = example['score']

        return x1, x2, score
