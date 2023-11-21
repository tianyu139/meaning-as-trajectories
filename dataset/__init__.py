# Meaning Representations from Trajectories in Autoregressive Models
# https://github.com/tianyu139/meaning-as-trajectories
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from functools import partial
from .CompareTask import CompareTask
from .STSB import STSB
from .STS12 import STS12
from .STS13 import STS13
from .STS14 import STS14
from .STS15 import STS15
from .STS16 import STS16
from .SICKR import SICKR
from .cxcsisbalanced import CXCSISBalanced
from .cxcsitsbalanced import CXCSITSBalanced
from .SNLIEntailment import SNLIEntailment

text_dataset_dict = {
    'stsbval': partial(STSB, test_set=False),
    'stsb': STSB,
    'sts12': STS12,
    'sts13': STS13,
    'sts14': STS14,
    'sts15': STS15,
    'sts16': STS16,
    'sickr': SICKR,
}

def get_text_dataset(dataset, *args, **kwargs) -> CompareTask:
    return text_dataset_dict[dataset](*args, **kwargs)

def get_image_dataset(*args, **kwargs) -> CompareTask:
    return CXCSISBalanced(*args, **kwargs)

def get_image_text_dataset(*args, **kwargs) -> CompareTask:
    return CXCSITSBalanced(*args, **kwargs)

def get_entailment_dataset(*args, **kwargs) -> CompareTask:
    return SNLIEntailment(*args, **kwargs)
