# Meaning Representations from Trajectories in Autoregressive Models
# https://github.com/tianyu139/meaning-as-trajectories
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from base_sampler import BaseSampler
import torch
from utils import get_trajectory_probabilities


class MultinomialSampler(BaseSampler):
    def __init__(self, n_traj, traj_len, tau):
        assert tau is not None
        self.tau = tau
        super().__init__(n_traj, traj_len)

    def sample(self, encoded_input, model, tokenizer, max_batch_size=10, images=None):
        kwargs = {}
        if images is not None:
            kwargs = {'images': images}
        with torch.no_grad():
            outputs = model.generate(
                inputs=encoded_input,
                do_sample=True,
                num_beams=1,
                num_return_sequences=self.n_traj,
                max_new_tokens=self.traj_len,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
                temperature=self.tau,
                **kwargs,
            )

        sequences = []
        for i in range(len(outputs.sequences)):
            sequence = outputs.sequences[i]
            sequence = sequence[len(encoded_input[0]):]
            for j, token in enumerate(sequence):
                if token == tokenizer.eos_token_id:
                    sequence = sequence[:j + 1]
            sequences.append(sequence)

        trajectories, _ = get_trajectory_probabilities(model, tokenizer, encoded_input, self.tau,
                            [s.unsqueeze(0) for s in sequences], images=images)
        return trajectories
