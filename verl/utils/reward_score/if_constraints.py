import os
import sys

sys.path.append(os.path.dirname(__file__))

import csv
import math
import random
import re
from difflib import SequenceMatcher
from typing import List, Union
from sentence_transformers import util, SentenceTransformer

import emoji
import numpy as np
import torch
import torch.nn.functional as F

from evaluation_lib import InputExample, test_instruction_following_strict
from instructions_util import split_into_sentences

# Global embedding model - lazy loaded and cached
_embedding_model = None

def get_embedding_model():
    """Lazy load embedding model (Ray-safe singleton pattern)."""
    global _embedding_model, _embedding_tokenizer
    
    if _embedding_model is None:
        model_name = "Qwen/Qwen3-Embedding-0.6B"
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"torch.cuda.device_count(): {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
        
        if device == "cuda":
            model_kwargs = {
                "attn_implementation": "flash_attention_2",
                "torch_dtype": torch.bfloat16
            }
        else:
            print("WARNING: GPU not available for embedding model, using CPU")
            model_kwargs = {
                "attn_implementation": "eager",
                "torch_dtype": torch.float32
            }
        
        _embedding_model = SentenceTransformer(
            model_name,
            model_kwargs=model_kwargs,
            tokenizer_kwargs={"padding_side": "left"},
            trust_remote_code=True,
            device=device
        ).eval()
        
        print(f"Loaded embedding model {model_name} on {device}")
    
    return _embedding_model


def compute_emb_similarity(texts: List[str]) -> torch.Tensor:
    """
    Compute embedding similarity for a list of texts.
    
    Args:
        texts: List of text strings to embed
    
    Returns:
        Tensor of shape (len(texts), len(texts)) with pairwise embedding similarity
    """
    model = get_embedding_model()
    embeddings = model.encode(texts)
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings)
    elif embeddings.is_cuda:
        embeddings = embeddings.cpu()
    sim_matrix = util.pytorch_cos_sim(embeddings, embeddings)
    
    return sim_matrix


def compute_diversity_scores(responses: List[str], threshold: float = 0.7) -> List[int]:
    """
    Compute diversity scores from a similarity matrix.

    Args:
        responses:
        threshold: Similarity threshold to consider for diversity (default is 0.7).

    Returns:
        List of diversity scores, one per element. Each score is the number of other elements
        with similarity less than the threshold.
    """
    sim_matrix = compute_emb_similarity(responses)
    
    mask = sim_matrix < threshold
    mask.fill_diagonal_(False)
    low_sim_counts = mask.sum(dim=1)
    N = sim_matrix.shape[0] - 1 if sim_matrix.shape[0] > 1 else 1
    diversity_scores = (low_sim_counts/N).tolist()

    return diversity_scores


def write_data(data, filename='/models/rewards.csv'):
    """
    Creates file with header if it doesn't exist, otherwise appends.
    
    Args:
        filename (str): Path to the output file
        data (list): List of tuples
    """
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['idx', 'reward_type', 'reward', 'split'])
        
        writer.writerows(data)


def follows_resp_format(text) -> bool:
    """Reward function that checks if the completion follows the strict format"""
    pattern = r"<response>.*\n</response>$"
    matches = re.search(pattern, text, re.DOTALL)
    return bool(matches)


def follows_tag_format(text: str, tag: str) -> bool:
    """
    Generalized reward function that penalizes format violations for any tag
    Returns: (is_valid, list_of_tag_contents)
    """
    
    opening_pattern = f"<{re.escape(tag)}>"
    closing_pattern = f"</{re.escape(tag)}>"
    
    opening_count = len(re.findall(opening_pattern, text))
    closing_count = len(re.findall(closing_pattern, text))
    
    if opening_count != closing_count or opening_count == 0:
        return False, []
    
    if text.count("<") != text.count(">"):
        return False, []
    
    opening_with_newline_pattern = f"<{re.escape(tag)}>\\n"
    if len(re.findall(opening_with_newline_pattern, text)) != opening_count:
        return False, []
    
    closing_with_newline_pattern = f"\\n</{re.escape(tag)}>"
    if len(re.findall(closing_with_newline_pattern, text)) != closing_count:
        return False, []
    
    stray_closing_pattern = f"</{re.escape(tag)}>\\s*$"
    complete_section_pattern = f"<{re.escape(tag)}>.*</{re.escape(tag)}>\\s*$"
    
    if re.search(stray_closing_pattern, text) and not re.search(complete_section_pattern, text, re.DOTALL):
        return False, []
    
    content_pattern = f"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>"
    tag_contents = re.findall(content_pattern, text, re.DOTALL)
    
    tag_contents = [content.strip() for content in tag_contents]
    
    return True, tag_contents


def thinking_len_reward(thinking, n_constraints):
    if not thinking:
        return 0
    rough_thinking_tokens = 1.3*len(thinking.split(' '))
    target = 256 #* (2 * math.log(n_constraints + 1))
    
    if rough_thinking_tokens < target:
        reward = rough_thinking_tokens / target
    else:
        penalty = (rough_thinking_tokens - target) / (target * 8)
        reward = max(0, 1 - penalty)
    return reward


def constraint_in_response(resp, constraints, constraint_types):
    if not constraints:
        return False
    copy_keywords = ['copy', 'repeat', 'reverse']
    if any(keyword in constraint.lower() for constraint in constraint_types for keyword in copy_keywords):
        return False
    resp_words = set(re.findall(r'\b\w+\b', resp.lower()))
    for constraint in constraints.split('\t'):
        constraint_words = [
            word for word in re.findall(r'\b\w+\b', constraint.lower())
            if len(word) >= 1
        ]
        pr_overlap = sum([1 for cw in constraint_words if cw in resp_words])
        if (pr_overlap / len(constraint_words)) >= 0.8:
            return True
    return False


def extract_xml_answer(text: str, tag: str) -> str:
    """Helper function to extract answers from XML format"""
    answer = text.split(f"<{tag}>")[-1]
    answer = answer.split(f"</{tag}>")[0]
    return answer.strip()


def is_fuzzy_pattern(text: str, constraint_types: List[str], threshold: float=0.5) -> bool:
    """
    Check if text contains fuzzy patterns by comparing consecutive elements.
    """
    
    def has_similar_consecutive_items(items, threshold):
        """Helper function to check for similar consecutive items in a list."""
        if len(items) == 1:
            is_emoji_list = [emoji.is_emoji(t) for t in items[0]]
            return any(is_emoji_list[i] and is_emoji_list[i+1] and is_emoji_list[i+2] for i in range(len(is_emoji_list)-2))
        for i in range(len(items) - 1):
            if len(items[i]) == 1 and len(items[i + 1]) == 1:
                return True
            if '\\' in items[i] or '\\' in items[i + 1]:
                return False
            seq_ratio = SequenceMatcher(None, items[i], items[i + 1]).ratio()
            if seq_ratio >= threshold:
                return True
        return False
    
    if len(text) < 3:
        return True
    
    if re.search(r'```|def\s+\w+\s*\(|class\s+\w+\s*[\(:]|function\s+\w+\s*\(|for\s+\w+\s+in\s+|if\s+.*:|import\s+\w+|from\s+\w+\s+import|\w+\s*=\s*\w+\s*\(|console\.log\(|print\s*\(', text):
        return False
    
    copy_keywords = ['copy', 'repeat', 'reverse']
    if any(keyword in constraint.lower() for constraint in constraint_types for keyword in copy_keywords):
        return False
    
    text_splits = [split_into_sentences(text), [t for t in text.split('\n') if t]]
    
    return any(has_similar_consecutive_items(split, threshold) for split in text_splits)


def max_length_normalized(binary_array, max_length=5, base=2):
    """Exponential reward weighting"""
    correct_count = sum(binary_array)
    if correct_count == 0:
        return 0
    max_possible_score = base ** max_length
    current_score = base ** correct_count
    return (current_score - 1) / (max_possible_score - 1)


def check_constraint_following(response, ground_truth, extra_info):
    instructs = ground_truth
    constraints = instructs["instruction_id"]
    eval_constraints = instructs['kwargs']
    prompt = 'na'
    inp = InputExample(
        key=0,
        instruction_id_list=constraints,
        prompt=prompt,
        kwargs=eval_constraints
    )
    constraint_eval = test_instruction_following_strict(
        inp, {prompt: response}
    )
    constraint_data = []
    for instr, follow_instr in zip(constraints, constraint_eval.follow_instruction_list):
        constraint_data.append((extra_info['index'], f'constr-{instr}', float(follow_instr), extra_info['split']))
    write_data(constraint_data)
    instr_level_reward = max_length_normalized(constraint_eval.follow_instruction_list, base=1.5)
    return instr_level_reward


def compute_score_single(solution_str, ground_truth, extra_info, data_source, diversity_score=0.0):
    """Score a single response with optional diversity bonus."""
    response = extract_xml_answer(solution_str, 'response')

    # Format rewards
    think_format, thoughts = follows_tag_format(solution_str, 'thinking')
    candidates_format, candidates = follows_tag_format(solution_str, 'candidate')
    resp_format = follows_resp_format(solution_str)
    
    if thoughts:
        think_long = np.mean([
            thinking_len_reward(thinking, len(ground_truth["instruction_id"]))
            for thinking in thoughts
        ])
    else:
        think_long = 0
    
    if candidates:
        candidate_long = np.mean([
            len(candidate) > 150 for candidate in candidates
        ])
    else:
        candidate_long = 0
    
    resp_long_enough = len(response) > 150
    
    format_reward = sum([
        0.10 if think_format else 0,
        think_long*0.6,
        0.10 if candidates_format else 0,
        candidate_long*0.1,
        0.10 if resp_format and resp_long_enough else 0,
    ])
    
    # Prevent reward hacking
    min_unique_words = len(set(response.split(' '))) > 10
    not_fuzzy_pattern = not is_fuzzy_pattern(response, ground_truth["instruction_id"], threshold=0.5)
    not_constraint_in_resp = not constraint_in_response(response, extra_info['constraints'], ground_truth["instruction_id"])
    no_hacking = min_unique_words and not_fuzzy_pattern and not_constraint_in_resp
    
    format_data = [
        (extra_info['index'], 'format-think_format', float(think_format), extra_info['split']),
        (extra_info['index'], 'format-candidates_formats', float(candidates_format), extra_info['split']),
        (extra_info['index'], 'format-resp_format', float(resp_format and resp_long_enough), extra_info['split']),
        (extra_info['index'], 'format-think_long', float(think_long), extra_info['split']),
        (extra_info['index'], 'format-candidate_long', float(candidate_long), extra_info['split']),
        (extra_info['index'], 'hack-min_unique_words', float(min_unique_words), extra_info['split']),
        (extra_info['index'], 'hack-not_fuzzy_pattern', float(not_fuzzy_pattern), extra_info['split']),
        (extra_info['index'], 'hack-not_constraint_in_resp', float(not_constraint_in_resp), extra_info['split']),
        (extra_info['index'], 'hack-no_hacking', float(no_hacking), extra_info['split']),
    ]
    write_data(format_data)
    
    # Constraint reward
    if extra_info['split'] == 'test':
        diversity_score = 1
    constraint_reward = check_constraint_following(response, ground_truth, extra_info)
    final_reward = diversity_score*(format_reward/10 + constraint_reward) if no_hacking else -1
    
    reward_data = [
        (extra_info['index'], 'train-constraint_reward', float(constraint_reward), extra_info['split']),
        (extra_info['index'], 'train-diversity_score', float(diversity_score), extra_info['split']),
        (extra_info['index'], 'train-final_reward', float(final_reward), extra_info['split']),
    ]
    write_data(reward_data)
    
    do_print = random.randint(1, 256) == 1 # print avg 4 per step
    if do_print:
        print(f"--------------------------------")
        print(f"final_reward: {final_reward}")
        print(f"constraint_reward: {constraint_reward} | format_reward: {format_reward} | diversity_score: {diversity_score}")
        print(f"think_format: {think_format} | candidates_format: {candidates_format} | resp_format: {resp_format} | think_long: {think_long} | candidate_long: {candidate_long} | resp_long_enough: {resp_long_enough}")
        print(f"min_unique_words: {min_unique_words} | not_fuzzy_pattern: {not_fuzzy_pattern} | not_constraint_in_resp: {not_constraint_in_resp} | no_hacking: {no_hacking}")
        print(f"{ground_truth} | constraint_text: {extra_info['constraints']}")
        print(f"[Solution string]\n{solution_str}")
        print(f"--------------------------------")
    
    return final_reward


def compute_score(solution_str, ground_truth, extra_info, data_source):
    """The scoring function for if task with diversity scoring.
    
    Handles both single responses and batched responses (lists).

    Args:
        solution_str: the solution text (str) OR list of solution texts
        ground_truth: dictionary OR list of dictionaries containing instruction info
        extra_info: extra info dict OR list of extra info dicts
        data_source: data source string OR list of data source strings
    """
    # Check if we're processing a batch (lists) or single item
    is_batch = isinstance(solution_str, list)
    
    if is_batch:
        # Extract responses for diversity computation
        responses = [extract_xml_answer(sol, 'response') for sol in solution_str]
        
        # Compute diversity scores for all responses in this batch
        diversity_scores = compute_diversity_scores(responses)
        
        # Process each item in the batch with its diversity score
        scores = []
        for sol, gt, ei, ds, div_score in zip(solution_str, ground_truth, extra_info, data_source, diversity_scores):
            score = compute_score_single(sol, gt, ei, ds, diversity_score=div_score)
            scores.append(score)
        return scores
    else:
        # Single item processing - no diversity bonus
        print('ERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRROR should not be here')
        return compute_score_single(solution_str, ground_truth, extra_info, data_source, diversity_score=1.0)