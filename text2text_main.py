# Meaning Representations from Trajectories in Autoregressive Models
# https://github.com/tianyu139/meaning-as-trajectories
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
import argparse
import time
import scipy.stats
from collections import defaultdict
import dataset
from dataset import get_text_dataset
from utils import *

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SUPPORTED_MODELS = ['gpt2', 'gpt2-xl', 'falcon7b', 'llama7b', 'llama13b',
                    'vicuna13b', 'stablevicuna13b',
                    'falcon40b', 'llama33b']


def evaluate_trajectory_similarity(sent1, sent2, model, tokenizer, traj_sampler, distance_metric, max_batch_size=10,
                                   pre_text_prompt='', post_text_prompt=''):
    sent1 = pre_text_prompt + sent1 + post_text_prompt
    sent2 = pre_text_prompt + sent2 + post_text_prompt
    encoded_sent1 = tokenizer.encode(sent1, return_tensors='pt').to(DEVICE)
    encoded_sent2 = tokenizer.encode(sent2, return_tensors='pt').to(DEVICE)

    encoded_sent1_trajs_and_probs = traj_sampler.sample(encoded_sent1, model, tokenizer, max_batch_size=max_batch_size)
    encoded_sent2_trajs_and_probs = traj_sampler.sample(encoded_sent2, model, tokenizer, max_batch_size=max_batch_size)

    encoded_sent1_trajs = [x.tokens.unsqueeze(0).to(DEVICE) for x in encoded_sent1_trajs_and_probs]
    sent1_traj_probs_with_sent1_prefix = [x.prob for x in encoded_sent1_trajs_and_probs]
    encoded_sent2_trajs = [x.tokens.unsqueeze(0).to(DEVICE) for x in encoded_sent2_trajs_and_probs]
    sent2_traj_probs_with_sent2_prefix = [x.prob for x in encoded_sent2_trajs_and_probs]

    stats_dict = {
        'sent1_trajs_text': [tokenizer.decode(traj[0]) for traj in encoded_sent1_trajs],
        'sent1_trajs_len': [len(traj[0]) for traj in encoded_sent1_trajs],
        'sent2_trajs_text': [tokenizer.decode(traj[0]) for traj in encoded_sent2_trajs],
        'sent2_trajs_len': [len(traj[0]) for traj in encoded_sent2_trajs],
    }

    _, sent2_traj_probs_with_sent1_prefix = get_trajectory_probabilities(model, tokenizer, encoded_sent1, traj_sampler.tau,
                                                                 encoded_choices=encoded_sent2_trajs, max_batch_size=max_batch_size)
    _, sent1_traj_probs_with_sent2_prefix = get_trajectory_probabilities(model, tokenizer, encoded_sent2, traj_sampler.tau,
                                                                 encoded_choices=encoded_sent1_trajs, max_batch_size=max_batch_size)

    all_traj_prob_with_sent1_prefix_prenorm = sent1_traj_probs_with_sent1_prefix + sent2_traj_probs_with_sent1_prefix
    all_traj_prob_with_sent2_prefix_prenorm = sent1_traj_probs_with_sent2_prefix + sent2_traj_probs_with_sent2_prefix

    # Do some post-processing of pre-normalized probabilities to get some preliminary statistics
    distance = distance_metric.compute(all_traj_prob_with_sent1_prefix_prenorm, all_traj_prob_with_sent2_prefix_prenorm)

    stats_dict['scores'] = -distance
    stats_dict['distances'] = distance
    stats_dict['all_traj_prob_with_sent1_prefix_prenorm'] = all_traj_prob_with_sent1_prefix_prenorm
    stats_dict['all_traj_prob_with_sent2_prefix_prenorm'] = all_traj_prob_with_sent2_prefix_prenorm

    # Lowest distance to highest

    return stats_dict


def main():
    parser = argparse.ArgumentParser(description='Semantic textual similarity.')
    # Common params
    parser.add_argument('model', type=str, help='model architecture', choices=SUPPORTED_MODELS)
    parser.add_argument('dataset', type=str, help='dataset',
                        choices=list(dataset.text_dataset_dict.keys()))
    parser.add_argument('compare_algo', type=str, help='comparison algorithm')
    parser.add_argument('--bs', type=int, help='maximum batch size to use for trajectories', default=100)
    parser.add_argument('--no_fullstop', action='store_true', help='end dataset examples with fullstop?')
    parser.add_argument('--save_output', type=str, help='where to save output', default='')
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    parser.add_argument('--debug_mode', action='store_true', help='debug mode evaluates only for one iter')
    parser.add_argument('--max_num_val', type=int, help='maximum size of validation set')
    parser.add_argument('--use_prompt_template', action='store_true', help='prompt template')
    # Trajectory Params
    parser.add_argument('--traj_algo', type=str, help='trajectory sampling algorithm', default='multinomial')
    parser.add_argument('--n_traj', type=int, help='how many trajectories to sample for open ended evaluation',
                        default=20)
    parser.add_argument('--max_traj_len', type=int,
                        help='maximum length of sampled trajectories for open ended evaluation', default=20)
    parser.add_argument('--tau', type=float, help='temperature to apply for sampling', default=1.0)
    parser.add_argument('--distance_metric', type=str, help='metric to use for computing distance between logprobs', default='l1')
    parser.add_argument('--distance_metric_temp', type=float, help='temperature for distance metric (applicable for TV, hellinger, etc.)', default=0.5)
    # Prompt params
    parser.add_argument('--pre_text_prompt', type=str, help='pre text prompt', default='')
    parser.add_argument('--post_text_prompt', type=str, help='post text prompt', default='')
    # End of params
    args = parser.parse_args()

    model, tokenizer = get_model(args.model)
    model.eval()

    max_data_num_val = args.max_num_val

    ds_val = get_text_dataset(args.dataset, is_train=False, max_data_num=max_data_num_val, end_with_fullstop=not args.no_fullstop)

    print(f"Val len: {len(ds_val)}")

    setup_seed(args.seed)

    metrics = {
        'accs': [],
        'spearman_corr': [],
        'gt_scores': [],
    }

    stats_dict_full = defaultdict(list)

    start_time = time.time()
    for ds_idx, data in enumerate(ds_val):
        sent1, sent2, gt_score = data

        if args.use_prompt_template:
            template = get_template(args.model)
            sent1 = template.format(query=sent1)
            sent2 = template.format(query=sent2)

        if args.compare_algo == 'trajectory':
            traj_sampler = get_traj_sampler(args)
            distance_metric = get_distance_metric(args.distance_metric, temp=args.distance_metric_temp)
            stats_dict = evaluate_trajectory_similarity(sent1, sent2, model, tokenizer, traj_sampler,
                                                             distance_metric=distance_metric,
                                                             max_batch_size=args.bs,
                                                             pre_text_prompt=args.pre_text_prompt,
                                                             post_text_prompt=args.post_text_prompt)
        else:
            raise ValueError("No such comparison algorithm")

        for k, v in stats_dict.items():
            stats_dict_full[k].append(v)

        scores = [s for s in stats_dict_full['scores']]
        metrics['gt_scores'].append(gt_score)
        spearman_corr = scipy.stats.spearmanr(scores, metrics['gt_scores']).correlation
        metrics['spearman_corr'] = spearman_corr
        running_metrics_print = f"Running spearman corr: {spearman_corr:.3f}"

        elapsed_time_per_epoch = (time.time() - start_time) / (ds_idx + 1)
        estimated_time_left = elapsed_time_per_epoch * (len(ds_val) - ds_idx - 1)
        print(
            f"[{ds_idx}/{len(ds_val)}]   {running_metrics_print}"
            f"  Avg Epoch Time: {int(elapsed_time_per_epoch)} sec   Est Time Left: {estimated_time_left / 60:.1f} min",
            end='\r', flush=True)

        if args.debug_mode:
            break

    print()

    final_spearman_corr = metrics['spearman_corr']
    print(
        f"[{len(ds_val)}/{len(ds_val)}]  Final spearman: {final_spearman_corr:.3f}")
    metrics['final_spearman_corr'] = final_spearman_corr

    results = {
        **vars(args),
        **stats_dict_full,
        **metrics,
    }

    if args.save_output:
        save_output(args, results)


if __name__ == '__main__':
    main()
