# Meaning Representations from Trajectories in Autoregressive Models
# https://github.com/tianyu139/meaning-as-trajectories
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
import argparse
import time
import numpy as np
from collections import defaultdict
from dataset import get_entailment_dataset
from utils import *

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SUPPORTED_MODELS = ['gpt2', 'gpt2-xl', 'falcon7b', 'instruct-falcon7b', 'llama13b']


def evaluate_trajectory_entailment(u, v, model, tokenizer, traj_sampler, distance_metric, max_batch_size=10):
    # Formulate u -> v as ( p(u and v), p(v) )  --> lower distance means higher implication score
    stats_dict = {}

    encoded_u = tokenizer.encode(u, return_tensors='pt').to(DEVICE)
    encoded_v = tokenizer.encode(v, return_tensors='pt').to(DEVICE)

    encoded_u_trajs_and_probs = traj_sampler.sample(encoded_u, model, tokenizer, max_batch_size=max_batch_size)
    encoded_u_trajs = [x.tokens.unsqueeze(0).to(DEVICE) for x in encoded_u_trajs_and_probs]
    u_traj_prob_with_u_prefix_prenorm = [x.prob for x in encoded_u_trajs_and_probs]

    encoded_v_trajs_and_probs = traj_sampler.sample(encoded_v, model, tokenizer, max_batch_size=max_batch_size)
    encoded_v_trajs = [x.tokens.unsqueeze(0).to(DEVICE) for x in encoded_v_trajs_and_probs]
    v_traj_prob_with_v_prefix_prenorm = [x.prob for x in encoded_v_trajs_and_probs]

    _, u_traj_prob_with_v_prefix_prenorm = get_trajectory_probabilities(model, tokenizer, encoded_v, traj_sampler.tau,
                                                                        encoded_choices=encoded_u_trajs,
                                                                        max_batch_size=max_batch_size)
    _, v_traj_prob_with_u_prefix_prenorm = get_trajectory_probabilities(model, tokenizer, encoded_u, traj_sampler.tau,
                                                                        encoded_choices=encoded_v_trajs,
                                                                        max_batch_size=max_batch_size)
    all_traj_prob_with_u_prefix_prenorm = u_traj_prob_with_u_prefix_prenorm + v_traj_prob_with_u_prefix_prenorm
    all_traj_prob_with_v_prefix_prenorm = u_traj_prob_with_v_prefix_prenorm + v_traj_prob_with_v_prefix_prenorm
    intersect_traj_prob_prenorm = [min([all_traj_prob_with_u_prefix_prenorm[i], all_traj_prob_with_v_prefix_prenorm[i]])
                                   for i in range(len(all_traj_prob_with_u_prefix_prenorm))]

    u_imply_v = distance_metric.compute(all_traj_prob_with_v_prefix_prenorm, intersect_traj_prob_prenorm)
    v_imply_u = distance_metric.compute(all_traj_prob_with_u_prefix_prenorm, intersect_traj_prob_prenorm)

    # u -> v if d (p(u and v), p(u) ) < d (p(u and v), p(v) )
    pred = 0 if u_imply_v < v_imply_u else 1

    stats_dict['u_trajs_text'] = [tokenizer.decode(traj[0]) for traj in encoded_u_trajs]
    stats_dict['u_trajs_len'] = [len(traj[0]) for traj in encoded_u_trajs]
    stats_dict['v_trajs_text'] = [tokenizer.decode(traj[0]) for traj in encoded_v_trajs]
    stats_dict['v_trajs_len'] = [len(traj[0]) for traj in encoded_v_trajs]
    stats_dict['u_imply_v'] = u_imply_v
    stats_dict['v_imply_u'] = v_imply_u
    stats_dict['all_traj_prob_with_u_prefix_prenorm'] = all_traj_prob_with_u_prefix_prenorm
    stats_dict['all_traj_prob_with_v_prefix_prenorm'] = all_traj_prob_with_v_prefix_prenorm
    stats_dict['pred'] = pred

    return pred, stats_dict


def main():
    parser = argparse.ArgumentParser(description='Evaluate asymmetric distances via SNLI Entailment dataset.')
    # Common params
    parser.add_argument('model', type=str, help='model architecture', choices=SUPPORTED_MODELS)
    parser.add_argument('compare_algo', type=str, help='comparison algorithm')
    parser.add_argument('--bs', type=int, help='maximum batch size to use for trajectories', default=10)
    parser.add_argument('--no_fullstop', action='store_true', help='end dataset examples with fullstop?')
    parser.add_argument('--save_output', type=str, help='where to save output', default='')
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    parser.add_argument('--debug_mode', action='store_true', help='debug mode evaluates only for one iter')
    parser.add_argument('--max_num_val', type=int, help='maximum size of validation set')
    # Trajectory Params
    parser.add_argument('--traj_algo', type=str, help='trajectory sampling algorithm', default='multinomial')
    parser.add_argument('--n_traj', type=int, help='how many trajectories to sample for open ended evaluation',
                        default=20)
    parser.add_argument('--max_traj_len', type=int,
                        help='maximum length of sampled trajectories for open ended evaluation', default=20)
    parser.add_argument('--tau', type=float, help='temperature to apply for sampling', default=1.0)
    parser.add_argument('--distance_metric', type=str, help='metric to use for computing distance between logprobs', default='l1')
    parser.add_argument('--distance_metric_temp', type=float, help='temperature for distance metric (applicable for TV, hellinger, etc.)', default=0.5)
    # End of params
    args = parser.parse_args()

    model, tokenizer = get_model(args.model)
    model.eval()

    max_data_num_val = args.max_num_val

    ds_val = get_entailment_dataset(is_train=False, max_data_num=max_data_num_val, end_with_fullstop=not args.no_fullstop)
    print(f"Val len: {len(ds_val)}")

    setup_seed(args.seed)

    metrics = {
        'accs': [],
        'gts': [],
    }

    stats_dict_full = defaultdict(list)

    start_time = time.time()
    for ds_idx, data in enumerate(ds_val):
        str_u, str_v, gt = data

        if args.compare_algo == 'trajectory':
            traj_sampler = get_traj_sampler(args)
            distance_metric = get_distance_metric(args.distance_metric, temp=args.distance_metric_temp)
            pred, stats_dict = evaluate_trajectory_entailment(str_u, str_v, model, tokenizer, traj_sampler,
                                                             distance_metric=distance_metric,
                                                             max_batch_size=args.bs)

        else:
            raise ValueError("No such comparison algorithm")

        for k, v in stats_dict.items():
            stats_dict_full[k].append(v)

        acc = pred == gt
        metrics['accs'].append(acc)
        metrics['gts'].append(gt)
        running_acc = np.mean(metrics['accs'])
        running_metrics_print = f"Running acc: {running_acc * 100:.2f}%"

        elapsed_time_per_epoch = (time.time() - start_time) / (ds_idx + 1)
        estimated_time_left = elapsed_time_per_epoch * (len(ds_val) - ds_idx - 1)
        print(
            f"[{ds_idx}/{len(ds_val)}]   {running_metrics_print}"
            f"  Avg Epoch Time: {int(elapsed_time_per_epoch)} sec   Est Time Left: {estimated_time_left / 60:.1f} min",
            end='\r', flush=True)

        if args.debug_mode:
            break

    print()
    final_acc = np.mean(metrics['accs'])
    print(f"[{len(ds_val)}/{len(ds_val)}]  Final acc: {final_acc * 100:.2f}%")
    metrics['final_acc'] = final_acc

    results = {
        **vars(args),
        **stats_dict_full,
        **metrics,
    }

    if args.save_output:
        save_output(args, results)


if __name__ == '__main__':
    main()
