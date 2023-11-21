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
from utils import *
from llava.mm_utils import tokenizer_image_token
from llava.constants import DEFAULT_IMAGE_TOKEN



DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SUPPORTED_MODELS = ['llava', 'llava_base']


def evaluate_trajectory_similarity(img_1, img_2, image_prompt_1, image_prompt_2, model, tokenizer, traj_sampler, distance_metric, max_batch_size=10):
    encoded_image_prompt_1 = tokenizer_image_token(image_prompt_1, tokenizer, return_tensors='pt').unsqueeze(0).to(DEVICE)
    encoded_image_prompt_2 = tokenizer_image_token(image_prompt_2, tokenizer, return_tensors='pt').unsqueeze(0).to(DEVICE)

    encoded_image_1_trajs_and_probs = traj_sampler.sample(encoded_image_prompt_1, model, tokenizer, max_batch_size=max_batch_size, images=img_1)
    encoded_image_2_trajs_and_probs = traj_sampler.sample(encoded_image_prompt_2, model, tokenizer, max_batch_size=max_batch_size, images=img_2)

    encoded_image_1_trajs = [x.tokens.unsqueeze(0).to(DEVICE) for x in encoded_image_1_trajs_and_probs]
    image_1_traj_probs_with_image_1_prefix = [x.prob for x in encoded_image_1_trajs_and_probs]

    encoded_image_2_trajs = [x.tokens.unsqueeze(0).to(DEVICE) for x in encoded_image_2_trajs_and_probs]
    image_2_traj_probs_with_image_2_prefix = [x.prob for x in encoded_image_2_trajs_and_probs]

    _, image_2_traj_probs_with_image_1_prefix = get_trajectory_probabilities(model, tokenizer, encoded_image_prompt_1,
                                                                           traj_sampler.tau,
                                                                           encoded_choices=encoded_image_2_trajs,
                                                                           max_batch_size=max_batch_size,
                                                                           images=img_1)
    _, image_1_traj_probs_with_image_2_prefix = get_trajectory_probabilities(model, tokenizer, encoded_image_prompt_2,
                                                                           traj_sampler.tau,
                                                                           encoded_choices=encoded_image_1_trajs,
                                                                           max_batch_size=max_batch_size,
                                                                           images=img_2)

    all_traj_prob_with_image_1_prefix_prenorm = image_1_traj_probs_with_image_1_prefix + image_2_traj_probs_with_image_1_prefix
    all_traj_prob_with_image_2_prefix_prenorm = image_1_traj_probs_with_image_2_prefix + image_2_traj_probs_with_image_2_prefix

    distance = distance_metric.compute(all_traj_prob_with_image_1_prefix_prenorm, all_traj_prob_with_image_2_prefix_prenorm)

    stats_dict = {
        'image_1_trajs_text': [tokenizer.decode(traj[0]) for traj in encoded_image_1_trajs],
        'image_1_trajs_len': [len(traj[0]) for traj in encoded_image_1_trajs],
        'image_2_trajs_text': [tokenizer.decode(traj[0]) for traj in encoded_image_2_trajs],
        'image_2_trajs_len': [len(traj[0]) for traj in encoded_image_2_trajs],
        'all_traj_prob_with_image_1_prefix_prenorm': all_traj_prob_with_image_1_prefix_prenorm,
        'all_traj_prob_with_image_2_prefix_prenorm': all_traj_prob_with_image_2_prefix_prenorm,
        'distances': distance,
        'scores': -distance,
    }

    return -distance, stats_dict


def main():
    parser = argparse.ArgumentParser(description='Semantic image-image similarity.')
    # Common params
    parser.add_argument('model', type=str, help='model architecture', choices=SUPPORTED_MODELS)
    parser.add_argument('compare_algo', type=str, help='comparison algorithm')
    parser.add_argument('--bs', type=int, help='maximum batch size to use for trajectories', default=50)
    parser.add_argument('--save_output', type=str, help='where to save output', default='')
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    parser.add_argument('--debug_mode', action='store_true', help='debug mode evaluates only for one iter')
    parser.add_argument('--max_num_val', type=int, help='maximum size of validation set', default=1000)
    # Prompt params
    parser.add_argument('--image_prompt', type=str, help='image prompt', default='')
    parser.add_argument('--post_image_prompt', type=str, help='post image prompt', default='')
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

    model, tokenizer, image_processor, conv_default = get_multimodal_model(args.model)
    assert not model.config.mm_use_im_start_end
    model.eval()

    max_data_num_val = args.max_num_val

    ds_val = dataset.get_image_dataset(max_data_num=max_data_num_val)
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
        img_raw_1, img_raw_2, gt_score = data

        img_1 = image_processor.preprocess(img_raw_1, return_tensors='pt')['pixel_values'].half().cuda()
        img_2 = image_processor.preprocess(img_raw_2, return_tensors='pt')['pixel_values'].half().cuda()
        conv_img_1 = conv_default.copy()
        conv_img_2 = conv_default.copy()

        if args.compare_algo == 'trajectory':
            prompt_img_1 = DEFAULT_IMAGE_TOKEN + '\n' + args.image_prompt
            prompt_img_2 = DEFAULT_IMAGE_TOKEN + '\n' + args.image_prompt

            conv_img_1.append_message(conv_img_1.roles[0], prompt_img_1)
            conv_img_1.append_message(conv_img_1.roles[1], None)
            conv_img_2.append_message(conv_img_2.roles[0], prompt_img_2)
            conv_img_2.append_message(conv_img_2.roles[1], None)
            prompt_img_1 = conv_img_1.get_prompt()
            prompt_img_2 = conv_img_2.get_prompt()
            prompt_img_1 += args.post_image_prompt
            prompt_img_2 += args.post_image_prompt

            traj_sampler = get_traj_sampler(args)
            distance_metric = get_distance_metric(args.distance_metric, temp=args.distance_metric_temp)
            _, stats_dict = evaluate_trajectory_similarity(img_1, img_2, prompt_img_1, prompt_img_2,
                                                                 model, tokenizer, traj_sampler,
                                                             distance_metric=distance_metric,
                                                             max_batch_size=args.bs)
        else:
            raise ValueError("No such comparison algorithm")

        for k, v in stats_dict.items():
            stats_dict_full[k].append(v)

        scores = stats_dict_full['scores']
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
