# Meaning Representations from Trajectories in Autoregressive Models
# https://github.com/tianyu139/meaning-as-trajectories
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
class BaseSampler:
    def __init__(self, n_traj, traj_len):
        assert n_traj > 0 and traj_len > 0
        self.n_traj = n_traj
        self.traj_len = traj_len
        self.transition_scores = None

    def sample(self, encoded_input, model, tokenizer, max_batch_size=10, images=None):
        pass







