# Meaning Representations from Trajectories in Autoregressive Models
# https://github.com/tianyu139/meaning-as-trajectories
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
import numpy as np
import random
import os
import pickle
from datetime import date
import scipy.special
import distance_metrics
from trajectory import Trajectory

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
LLAMA_7B_PATH = './pretrained/LLaMa/7B'
LLAMA_13B_PATH = './pretrained/LLaMa/13B'
LLAMA_33B_PATH = './pretrained/LLaMa/33B'

def get_model(model_name):
    if model_name == 'gpt2-xl':
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained('gpt2-xl', device_map='auto')
    elif model_name == 'gpt2':
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained('gpt2', device_map='auto')
    elif model_name == 'llama7b':
        from transformers import LlamaForCausalLM, LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(LLAMA_7B_PATH)
        model = LlamaForCausalLM.from_pretrained(LLAMA_7B_PATH, device_map="auto")
        model.half()
    elif model_name == 'llama13b':
        from transformers import LlamaForCausalLM, LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(LLAMA_13B_PATH)
        model = LlamaForCausalLM.from_pretrained(LLAMA_13B_PATH, device_map="auto")
        model.half()
    elif model_name == 'llama33b':
        from transformers import LlamaForCausalLM, LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(LLAMA_33B_PATH)
        model = LlamaForCausalLM.from_pretrained(LLAMA_33B_PATH, device_map="auto")
        model.half()
    elif model_name == 'falcon7b':
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model = "tiiuae/falcon-7b"
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                     device_map="auto")
    elif model_name == 'falcon40b':
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model = "tiiuae/falcon-40b"
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                     device_map="auto")
    elif model_name == 'instruct-falcon7b':
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model = "tiiuae/falcon-7b-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                     device_map="auto")
    else:
        raise ValueError("Model not supported")

    if tokenizer is not None and tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def get_multimodal_model(model_name):
    if model_name == 'llava':
        import llava.model.builder
        import llava.conversation
        # Changes langauge model
        model = 'liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3'
        tokenizer, model, image_processor, _ = llava.model.builder.load_pretrained_model(model, None, model)
        assert model.config.mm_use_im_patch_token is False and model.config.mm_use_im_start_end is False
        conv_mode = "llava_v1"
        conv = llava.conversation.conv_templates[conv_mode].copy()
    else:
        raise ValueError("Model not supported")

    return model, tokenizer, image_processor, conv

def get_traj_sampler(args):
    algo = args.traj_algo
    if algo == 'multinomial':
        from multinomial_sampler import MultinomialSampler
        return MultinomialSampler(args.n_traj, args.max_traj_len, args.tau)
    else:
        raise ValueError("No such sampler")


def softmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x."""
    x = np.array(x)
    return scipy.special.softmax(x, axis=axis)


def get_distance_metric(name, temp=1.0):
    if name == 'l1':
        return distance_metrics.L1Distance()
    elif name == 'l2':
        return distance_metrics.L2Distance()
    elif name == 'linf':
        return distance_metrics.LInfDistance()
    elif name == 'symkl':
        return distance_metrics.SymmetricKLDivergence(temp=temp)
    elif name == 'tv':
        return distance_metrics.TotalVariationDistance(temp=temp)
    elif name == 'hellinger':
        return distance_metrics.HellingerDistance(temp=temp)
    elif name == 'cossim':
        return distance_metrics.CosineSimilarity()
    elif name == 'cossim-prob':
        return distance_metrics.CosineSimilarityProb(temp=temp)
    else:
        raise ValueError("Distance metric not supported")


def get_template(model):
    if model == 'stablevicuna13b':
        template = "### Human: {query}\n### Assistant: "
    elif model == 'vicuna13b':
        template = "A chat between a curious user and an artificial intelligence assistant. " + \
            "The assistant gives helpful, detailed, and polite answers to the user's questions.\n" + \
            "USER: {query} ASSISTANT:"
    else:
        raise ValueError("Template not supported for this model")

    return template

def pad_to_length(x, pad_length, pad_token):
    # Assume x is of shape (1,n)
    pad_token = torch.tensor(pad_token).to(x.device)
    new_tokens = pad_token.unsqueeze(0).repeat(1, pad_length - x.shape[1])
    return torch.cat([x, new_tokens], dim=1)

@torch.no_grad()
def get_trajectory_probabilities(model,
                     tokenizer,
                     encoded_query,
                     tau,
                     encoded_choices=None,
                     normalize_length=True,
                     max_batch_size=10,
                     images=None):
    assert normalize_length is True and encoded_choices is not None

    trajectories = []
    n_batches = ((len(encoded_choices) - 1) // max_batch_size) + 1
    encoded_choices_idx_split = np.array_split(list(range(len(encoded_choices))), n_batches)
    for encoded_choices_idx_batch in encoded_choices_idx_split:
        encoded_choices_batch = [encoded_choices[idx] for idx in encoded_choices_idx_batch]
        encoded_choice_len = [len(c[0]) for c in encoded_choices_batch]
        max_len_to_pad = max([len(c[0]) for c in encoded_choices_batch])
        encoded_choices_padded = [pad_to_length(c, max_len_to_pad, tokenizer.pad_token_id) for c in encoded_choices_batch]
        encoded_choices_padded = torch.cat(encoded_choices_padded, dim=0)
        encoded_input_batch = torch.cat([encoded_query.repeat(len(encoded_choices_batch), 1), encoded_choices_padded], dim=1)
        if images is not None:
            images = torch.cat([images] * len(encoded_choices_batch), dim=0)
            outputs_batch = model(encoded_input_batch, images=images)
        else:
            outputs_batch = model(encoded_input_batch)
        last_idx = outputs_batch.logits.shape[1] - encoded_choices_padded.shape[1] - 1
        choice_batch_logits = outputs_batch.logits[:, last_idx:-1]
        choice_batch_probs = torch.log_softmax(choice_batch_logits / tau, dim=-1)

        for choice_len, choice_prob, encoded_choice in zip(encoded_choice_len, choice_batch_probs, encoded_choices_batch):
            traj_tokens_probs = choice_prob[torch.arange(choice_len).to(DEVICE), encoded_choice.flatten()]
            if normalize_length:
                reduction = 'mean'
            else:
                reduction = 'sum'

            trajectories.append(Trajectory(encoded_choice.flatten(), traj_tokens_probs, reduction=reduction))

    return trajectories, [t.prob for t in trajectories]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_output(args, results):
    output_folder = "outputs"
    output_folder = os.path.join(output_folder, args.save_output + '_' + date.today().strftime("%d-%m-%y"))
    if hasattr(args, 'model1'):
        output_folder = os.path.join(output_folder, f"{args.model1}-{args.model2}-{args.dataset}-seed={args.seed}")
    elif hasattr(args, 'compare_algo'):
        output_folder = os.path.join(output_folder, f"{args.model}-{args.dataset}-{args.compare_algo}-seed={args.seed}")
    else:
        output_folder = os.path.join(output_folder, f"{args.model}-{args.dataset}-seed={args.seed}")
    number = 0
    output_basefile = f"{number}.pickle"
    output_file = os.path.join(output_folder, output_basefile)
    while os.path.exists(output_file):
        number += 1
        output_basefile = f"{number}.pickle"
        output_file = os.path.join(output_folder, output_basefile)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"Saved results to: {output_file}")