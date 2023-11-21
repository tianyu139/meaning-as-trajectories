# Meaning Representations from Trajectories in Autoregressive Models
# https://github.com/tianyu139/meaning-as-trajectories
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
class Trajectory:
    def __init__(self, tokens, token_probs=None, reduction='mean'):
        self.tokens = tokens
        self.token_probs = token_probs
        self.reduction = reduction
        if reduction == 'mean':
            self.prob = token_probs.mean().item()
        elif reduction == 'sum':
            self.prob = token_probs.sum().item()
        else:
            raise ValueError(f"Unhandled reduction: {reduction}")
