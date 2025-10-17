import os
import sys

sys.path.append(os.path.dirname(__file__))

import csv
import json
import math
import random
import re
from difflib import SequenceMatcher
from typing import Dict, List, Union

import emoji
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer, util

from evaluation_lib import InputExample, test_instruction_following_strict
from instructions_util import split_into_sentences


class ConstraintMatcher:
    """Rule-based constraint matching system with Hungarian algorithm matching."""
    
    def __init__(self):
        # Define keyword patterns for each constraint (using actual candidate names)
        self.patterns = {
            # Keywords group
            'keywords:word_once': [
                r'\binclude\s+keyword\b(?!.*\d+\s*times?)',
                r'\bkeyword\b.*\bin\s+(your\s+)?response\b',
                r'\buse\s+keyword\b(?!.*\d+\s*times?)'
            ],
            'keywords:word_count_different_numbers': [
                r'\b(word|keyword)\b.*\bshould\s+appear\b.*\b\d+\s*times?\b',
                r'\b(word|keyword)\b.*\bappear\b.*\b\d+\s*times?\b',
                r'\bin\s+your\s+response,?\s+the\s+(word|keyword)\b.*\b\d+\s*times?\b'
            ],
            'keywords:exclude_word_harder': [
                r'\bdo\s+not\s+include\s+(keyword|word)\b',
                r'\bavoid\s+(the\s+)?(word|keyword)\b',
                r'\bexclude\s+(the\s+)?(word|keyword)\b',
                r'\bdon\'?t\s+use\s+(the\s+)?(keyword|word)\b'
            ],
            'keywords:existence': [
                r'\binclude\s+keywords?\b',
                r'\bkeywords?\b.*\bin\s+(your\s+)?response\b',
                r'\bmention\s+keywords?\b'
            ],
            'keywords:forbidden_words': [
                r'\bdo\s+not\s+include\s+keywords?\b',
                r'\bforbidden\s+words?\b',
                r'\bavoid\s+keywords?\b'
            ],
            'keywords:frequency': [
                r'\b(word|keyword)\b.*\bappear\b.*\b\d+\s*times?\b',
                r'\b(word|keyword)\b.*\bfrequency\b'
            ],
            'keywords:keyword_specific_position': [
                r'\binclude\s+(keyword|word)\b.*\bin\s+the\s+\d+(th|st|nd|rd)?\s*(-\s*th)?\s+sentence\b',
                r'\b(as\s+the\s+)?\d+(th|st|nd|rd|m-th)?\s+word\b',
                r'\bsentence\b.*\bword\b.*\bposition\b'
            ],
            'keywords:letter_frequency': [
                r'\bletter\b.*\bshould\s+appear\b.*\b\d+\s*times?\b',
                r'\bletter\b.*\bfrequency\b',
                r'\bin\s+your\s+response,?\s+the\s+letter\b.*\b\d+\s*times?\b'
            ],
            'keywords:palindrome': [
                r'\bpalindrome\b',
                r'\binclude\s+a\s+palindrome\b'
            ],
            'keywords:start_end': [
                r'\bstart\s+and\s+end\b.*\b(with\s+the\s+)?same\s+word\b',
                r'\bbegin\s+and\s+end\b.*\bsame\s+word\b',
                r'\bfirst\s+and\s+last\s+word\b.*\bsame\b'
            ],
            'keywords:no_adjacent_consecutive': [
                r'\bno\s+two\s+adjacent\s+words?\b.*\bconsecutive\s+letters?\b',
                r'\badjacent\s+words?\b.*\bcan\'?t\s+start\b.*\bconsecutive\b',
                r'\bconsecutive\s+letters\s+of\s+the\s+alphabet\b'
            ],
            
            # Letters group
            'letters:letter_counting': [
                r'\banswer\s+with\s+(at\s+(least|most)|around|exactly)?\s*\d+\s+letters?\b',
                r'\bexactly\s+\d+\s+letters?\b',
                r'\b(at\s+(least|most)|around)\s+\d+\s+letters?\b'
            ],
            'letters:letter_counting2': [
                r'\bletter\b.*\bshould\s+appear\b.*\b\d+\s*times?\b',
                r'\bin\s+your\s+response,?\s+the\s+letter\b.*\b\d+\s*times?\b'
            ],
            
            # Paragraphs group
            'paragraphs:paragraphs': [
                r'\bcontain\s+\d+\s+paragraphs?\b.*\bmarkdown\s+divider\b',
                r'\bmarkdown\s+divider\s*:\s*\*\s*\*\s*\*',
                r'\bseparate\s+paragraphs?\b.*\bmarkdown\b'
            ],
            'paragraphs:paragraphs2': [
                r'\b\d+\s+paragraphs?\b.*\bseparated\b.*\btwo\s+line\s+breaks?\b',
                r'\bparagraphs?\s+and\s+only\s+paragraphs?\b.*\bseparated\b.*\btwo\s+line\s+breaks?\b',
                r'\btwo\s+line\s+breaks?\b'
            ],
            
            # First word group
            'first_word:first_word_sent': [
                r'\bfirst\s+word\s+of\s+each\s+sentence\b',
                r'\bstart\s+each\s+sentence\b.*\bword\b',
                r'\beach\s+sentence\b.*\bshould\s+start\b'
            ],
            'first_word:first_word_answer': [
                r'\bfirst\s+word\s+of\s+(your\s+)?response\b',
                r'\bstart\s+(your\s+)?response\b.*\bword\b',
                r'\bbegin\s+(your\s+)?response\b.*\bword\b'
            ],
            
            # Last word group
            'last_word:last_word_sent': [
                r'\blast\s+word\s+of\s+each\s+sentence\b',
                r'\bend\s+each\s+sentence\b.*\bword\b',
                r'\beach\s+sentence\b.*\bshould\s+end\b'
            ],
            'last_word:last_word_answer': [
                r'\blast\s+word\s+of\s+(your\s+)?response\b',
                r'\bend\s+(your\s+)?response\b.*\bword\b',
                r'\bfinish\s+(your\s+)?response\b.*\bword\b'
            ],
            
            # Detectable format group
            'detectable_format:bigram_wrapping': [
                r'\bwrap\s+(every\s+)?word\s+bigram\b',
                r'\bdouble\s+angular\s+brackets?\b',
                r'«.*»'
            ],
            'detectable_format:square_brackets': [
                r'\benclose\s+every\s+word\b.*\bsquare\s+brackets?\b',
                r'\bwrap\b.*\bsquare\s+brackets?\b',
                r'\bwithin\s+square\s+brackets?\b'
            ],
            'detectable_format:sentence_hyphens': [
                r'\bsentences?\b.*\bconnected\b.*\bhyphens?\b',
                r'\bhyphens?\b.*\bno\s+spaces?\b',
                r'\ball\s+sentences?\s+must\s+be\s+connected\b'
            ],
            'detectable_format:number_bullet_lists': [
                r'\b(contain\s+)?(exactly\s+)?\d+\s+bullet\s+points?\b',
                r'\bmarkdown\s+bullet\s+points?\b',
                r'\*\s+[Tt]his\s+is\s+a\s+point'
            ],
            'detectable_format:title': [
                r'\btitle\b.*\bdouble\s+angular\s+brackets?\b',
                r'\bwrapped\s+in\s+double\s+angular\s+brackets?\b',
                r'<<.*>>'
            ],
            'detectable_format:constrained_response': [
                r'\banswer\s+with\s+one\s+of\s+the\s+following\b',
                r'\bchoose\s+from\b',
                r'\bone\s+of\s+the\s+following\s+options?\b'
            ],
            'detectable_format:number_highlighted_sections': [
                r'\bhighlight\s+(at\s+least\s+)?\d+\s+sections?\b',
                r'\bmarkdown\b.*\*.*\*',
                r'\bhighlighted\s+section\b'
            ],
            'detectable_format:multiple_sections': [
                r'\bmust\s+have\s+\d+\s+sections?\b',
                r'\bmark\s+the\s+beginning\s+of\s+each\s+section\b',
                r'\bsection\s+splitter\b'
            ],
            'detectable_format:json_format': [
                r'\bJSON\s+format\b',
                r'\bwrapped\s+in\s+JSON\b',
                r'\bentire\s+output\b.*\bJSON\b'
            ],
            
            # Copy group
            'copy:copying_simple': [
                r'\brepeat\s+the\s+request\b.*\bwithout\s+change\b(?!.*\d+\s*times?)',
                r'\bdo\s+not\s+answer\b.*\bactual\s+request\b',
                r'\brepeat\b.*\brequest\b(?!.*\d+\s*times?).*\bdo\s+not\s+answer\b'
            ],
            'copy:copying_multiple': [
                r'\brepeat\s+the\s+request\b.*\b\d+\s*times?\b',
                r'\bseparated\s+by\s+6\s+asterisk',
                r'\brepeat\b.*\bwithout\s+change\b.*\b\d+\s*times?\b'
            ],
            'copy:copy': [
                r'\bcopy\b.*\bverbatim\b',
                r'\bcopy\s+this\s+instruction\b',
                r'\bdo\s+not\s+follow\s+the\s+instruction\b'
            ],
            'copy:copy_span_idx': [
                r'\bcopy\s+the\s+span\b',
                r'\bindex\s+\d+\s+and\s+\d+\b',
                r'\bcharacter\s+indices?\b',
                r'\bbetween.*including.*index\b'
            ],
            'copy:repeat_phrase': [
                r'\brepeat\s+the\s+phrase\b.*\b\d+\s*times?\b',
                r'\btransforming\b.*\breplacing\b',
                r'\brepeat\b.*\bexactly\s+\d+\s*times?\b'
            ],
            
            # Punctuation group
            'punctuation:punctuation_dot': [
                r'\brefrain\b.*\b(use\s+of\s+)?\.',
                r'\bno\s+dots?\b',
                r'\bavoid.*\bperiods?\b',
                r'\brefrain.*dots?\b'
            ],
            'punctuation:punctuation_exclamation': [
                r'\brefrain\b.*\bexclamation\s+marks?\b',
                r'\bno\s+!\b',
                r'\bavoid.*\bexclamation\b',
                r'\brefrain.*!\b'
            ],
            'punctuation:no_comma': [
                r'\brefrain.*\bcommas?\b',
                r'\bno\s+commas?\b',
                r'\bavoid.*\bcommas?\b',
                r'\brefrain.*\buse\s+of\s+(any\s+)?commas?\b'
            ],
            
            # Count group
            'count:lowercase_counting': [
                r'\blowercase\s+words?\b.*\bat\s+most\s+\d+\s*times?\b',
                r'\ball\s+lowercase\s+words?\b.*\bappear\s+at\s+most\b'
            ],
            'count:letter_counting': [
                r'\banswer\s+with\s+(exactly\s+)?\d+\s+letters?\b',
                r'\brelation\s+\d+\s+letters?\b'
            ],
            'count:counting_composition': [
                r'\b\d+\s+paragraphs?\b.*\bexactly\s+\d+\s+sentences?\b',
                r'\bexactly\s+\d+\s+words?\s+in\s+each\s+sentence\b',
                r'\bdelimited\s+by\s+the\s+markdown\s+divider\b'
            ],
            'count:count_unique': [
                r'\bonly\s+use\s+unique\s+words?\b',
                r'\bno\s+word\s+should\s+be\s+repeated\b',
                r'\bno\s+repetition\b'
            ],
            'count:count_increment_word': [
                r'\bkeyword\b.*\bonce\b.*\bkeyword\b.*\btwice\b',
                r'\bonce\s+in\s+your\s+response\b.*\btwice\s+in\s+your\s+response\b'
            ],
            
            # Language group
            'language:response_language': [
                r'\bentire\s+response\b.*\bin\s+\w+\b',
                r'\bresponse\s+should\s+be\s+in\s+\w+\b',
                r'\byour\s+ENTIRE\s+response\b.*\blanguage\b'
            ],
            
            # Length Constraints
            'length_constraints:number_paragraphs': [
                r'\bcontain\s+\d+\s+paragraphs?\b',
                r'\bshould\s+contain\s+\d+\s+paragraphs?\b',
                r'\bmarkdown\s+divider\b'
            ],
            'length_constraints:number_words': [
                r'\banswer\s+with\s+(at\s+least|around|at\s+most)\s+\d+\s+words?\b',
                r'\b(at\s+least|around|at\s+most)\s+\d+\s+words?\b'
            ],
            'length_constraints:number_sentences': [
                r'\banswer\s+with\s+(at\s+least|around|at\s+most)\s+\d+\s+sentences?\b',
                r'\b(at\s+least|around|at\s+most)\s+\d+\s+sentences?\b'
            ],
            'length_constraints:nth_paragraph_first_word': [
                r'\bparagraphs?\b.*\bmust\s+start\s+with\s+word\b',
                r'\b\d+(th|st|nd|rd)\s+paragraph\b.*\bfirst\s+word\b',
                r'\bi-th\s+paragraph\b'
            ],
            
            # Detectable Content
            'detectable_content:postscript': [
                r'\bpostscript\b',
                r'\bP\.?\s*S\.?\b',
                r'\bat\s+the\s+end\b.*\bexplicitly\s+add\b.*\bpostscript\b'
            ],
            'detectable_content:number_placeholders': [
                r'\bplaceholders?\b.*\bsquare\s+brackets?\b',
                r'\bat\s+least\s+\d+\s+placeholders?\b',
                r'\brepresented\s+by\s+square\s+brackets?\b'
            ],
            
            # Combination
            'combination:repeat_prompt': [
                r'\bfirst,?\s+repeat\b.*\bthen\s+give\b',
                r'\brepeat\s+the\s+request\b.*\bthen\s+give\s+your\s+answer\b'
            ],
            'combination:two_responses': [
                r'\bgive\s+two\s+different\s+responses?\b',
                r'\bseparated\s+by\s+6\s+asterisk\b',
                r'\*\*\*\*\*\*'
            ],
            
            # Change Cases
            'change_case:english_capital': [
                r'\ball\s+uppercase\b',
                r'\bcapital\s+letters?\s+only\b',
                r'\bentire\s+response\b.*\bEnglish,?\s+capital\s+letters?\b'
            ],
            'change_case:english_lowercase': [
                r'\ball\s+lowercase\b',
                r'\blowercase\s+letters?\s+only\b',
                r'\bno\s+capital\s+letters?\b'
            ],
            'change_case:capital_word_frequency': [
                r'\bwords?\s+with\s+all\s+capital\s+letters?\b',
                r'\ball\s+capital\s+letters?\b.*\bat\s+(least|most)\b',
                r'\bcapital\s+word\b.*\bfrequency\b'
            ],
            
            # Start/End
            'startend:end_checker': [
                r'\bfinish\b.*\bwith\b.*\bexact\s+phrase\b',
                r'\bend\b.*\bexact\s+phrase\b',
                r'\bno\s+other\s+words?\s+should\s+follow\b'
            ],
            'startend:quotation': [
                r'\bwrap\b.*\bentire\s+response\b.*\bquotation\s+marks?\b',
                r'\bdouble\s+quotation\s+marks?\b',
                r'\bquotation\s+marks?\b.*\bentire\s+response\b'
            ]
        }

    # Helper function to match instruction to pattern
    def match_instruction(instruction_text):
        """
        Match an instruction text to the appropriate pattern category.
        Returns a list of matching pattern keys.
        """
        instruction_lower = instruction_text.lower()
        matches = []
        
        for pattern_key, pattern_list in self.patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, instruction_lower, re.IGNORECASE):
                    matches.append(pattern_key)
                    break
        
        return matches
        
    def calculate_similarity(self, line: str, constraint: str) -> float:
        """
        Calculate similarity score between a line and a constraint.
        
        Args:
            line: A single line from the verification text
            constraint: A constraint name
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if constraint not in self.patterns:
            print('EEEEEEEEEEEEEEEEEEEE')
            return 0.0
        
        patterns = self.patterns[constraint]
        matches = 0
        
        for pattern in patterns:
            if re.search(pattern, line, re.IGNORECASE):
                matches += 1
        
        # Normalize by number of patterns
        score = matches / len(patterns) if patterns else 0.0
        
        # Bonus for exact/strong matches
        if matches > 0:
            score = min(1.0, score * 1.5)
        
        return score
    
    def create_cost_matrix(self, lines: List[str], candidates: List[str]) -> np.ndarray:
        """
        Create cost matrix for Hungarian algorithm.
        
        Args:
            lines: List of verification lines
            candidates: List of candidate constraint names
            
        Returns:
            Cost matrix (lines x candidates), where lower cost = better match
        """
        n_lines = len(lines)
        n_candidates = len(candidates)
        
        # Create similarity matrix
        similarity_matrix = np.zeros((n_lines, n_candidates))
        
        for i, line in enumerate(lines):
            for j, candidate in enumerate(candidates):
                similarity_matrix[i, j] = self.calculate_similarity(line, candidate)
        
        # Convert similarity to cost (higher similarity = lower cost)
        # Use 1 - similarity as cost
        cost_matrix = 1.0 - similarity_matrix

        print(similarity_matrix)
        
        return cost_matrix, similarity_matrix
    
    def lines_to_constraint(self, lines: List[str], candidates: List[str]) -> Dict[str, int]:
        """
        Map lines to constraints using Hungarian algorithm and extract judgments.
        
        Args:
            lines: List of verification lines (already lowercased)
            candidates: List of candidate constraint names
            
        Returns:
            Dictionary mapping each candidate to 1 (true) or 0 (false/missing)
        """
        # Initialize result with all candidates set to 0
        result = {candidate: 0 for candidate in candidates}
        
        # Filter out empty lines
        non_empty_lines = [line for line in lines if line.strip()]
        
        if not non_empty_lines:
            return result
        
        # Create cost matrix
        cost_matrix, similarity_matrix = self.create_cost_matrix(non_empty_lines, candidates)
        
        # Apply Hungarian algorithm
        # Note: we need to handle the case where len(lines) != len(candidates)
        n_lines = len(non_empty_lines)
        n_candidates = len(candidates)
        
        if n_lines == 0:
            return result
        
        # Pad matrix if needed to make it square
        if n_lines < n_candidates:
            # More candidates than lines - pad with high cost rows
            padding = np.ones((n_candidates - n_lines, n_candidates))
            cost_matrix = np.vstack([cost_matrix, padding])
            similarity_matrix = np.vstack([similarity_matrix, np.zeros((n_candidates - n_lines, n_candidates))])
        elif n_lines > n_candidates:
            # More lines than candidates - pad with high cost columns
            padding = np.ones((n_lines, n_lines - n_candidates))
            cost_matrix = np.hstack([cost_matrix, padding])
            similarity_matrix = np.hstack([similarity_matrix, np.zeros((n_lines, n_lines - n_candidates))])
        
        # Run Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Extract matches and determine true/false
        for line_idx, candidate_idx in zip(row_ind, col_ind):
            # Only consider matches within the original dimensions
            if line_idx < n_lines and candidate_idx < n_candidates:
                candidate = candidates[candidate_idx]
                line = non_empty_lines[line_idx]
                similarity = similarity_matrix[line_idx, candidate_idx]
                # Only accept match if similarity is above threshold
                if similarity > 0.1:  # Minimum threshold for valid match
                    print(line_idx, candidate_idx)
                    # Check if line contains 'true' or positive indicators
                    if re.search(r'\btrue\b', line, re.IGNORECASE):
                        result[candidate] = 'true'
                    elif re.search(r'\bunsure\b|\bunclear\b|\bmaybe\b|\bpossibly\b|\bunknown\b', line, re.IGNORECASE):
                        result[candidate] = 'unsure'
                    elif re.search(r'\bfalse\b|\bfail', line, re.IGNORECASE):
                        result[candidate] = 'false'
                    else:
                        # Default to 1 if matched with good similarity but no explicit true/false
                        result[candidate] = 'nm'
        
        return result
    
    def extract_verify_logic(self, verify: str, candidates: List[str]) -> Dict[str, int]:
        """
        Main function to extract verification logic from verify text.
        
        Args:
            verify: The verification text with multiple lines
            candidates: List of candidate constraint names
            
        Returns:
            Dictionary mapping each candidate to 1 (true) or 0 (false/missing)
        """
        lines = [line.lower() for line in verify.split('\n')]
        verify_scores = self.lines_to_constraint(lines, candidates)
        return verify_scores


# Standalone function for easy import
def extract_verify_logic(verify: str, candidates: List[str]) -> Dict[str, int]:
    """
    Maps the verification section to atomic constraints with the judgment.
    
    Args:
        verify: The verification text with multiple lines
        candidates: List of candidate constraint names
        
    Returns:
        Dictionary mapping each candidate to 1 (true) or 0 (false/missing)
    """
    matcher = ConstraintMatcher()
    return matcher.extract_verify_logic(verify, candidates)


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


def compute_diversity_scores(responses: List[str], threshold: float = 0.8) -> List[int]:
    """
    Compute diversity scores from a similarity matrix.

    Args:
        responses:
        threshold: Similarity threshold to consider for diversity (default is 0.8).

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

def write_data_jsonl(data, filename='/models/rewards_mt.jsonl'):
    """
    Appends Python dict(s) to a JSONL file, creating the file if it doesn't exist.
    
    Args:
        filename (str): Path to the output JSONL file
        data (dict or list): Single dict or list of dicts to append
    """
    # Ensure data is a list
    if isinstance(data, dict):
        data = [data]
    
    with open(filename, 'a', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


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
    target = 256 * (1.5 * math.log(n_constraints + 1)) # scaled to number of constraints
    # target = 512
    
    if rough_thinking_tokens < target:
        reward = rough_thinking_tokens / target
    else:
        penalty = (rough_thinking_tokens - target) / (target * 4)
        reward = max(0, 1 - penalty)
    return reward


def constraint_in_response(resp, constraints, constraint_types):
    if not constraints:
        return False
    copy_keywords = ['copy', 'repeat', 'reverse', 'constrained_response']
    if any(keyword in constraint.lower() for constraint in constraint_types for keyword in copy_keywords):
        return False
    resp_words = set(re.findall(r'\b\w+\b', resp.lower()))
    for constraint in constraints.split('\t'):
        constraint_words = [
            word for word in re.findall(r'\b\w+\b', constraint.lower())
            if len(word) >= 1
        ]
        pr_overlap = sum([1 for cw in constraint_words if cw in resp_words])
        if (pr_overlap / len(constraint_words)) >= 0.7:
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
            # if len(items[i]) == 1 and len(items[i + 1]) == 1:
            #     return True
            if '\\' in items[i] or '\\' in items[i + 1]:
                return False
            seq_ratio = SequenceMatcher(None, items[i], items[i + 1]).ratio()
            if seq_ratio >= threshold:
                return True
        return False
    
    # if len(text) < 3:
    #     return True
    
    if re.search(r'```|def\s+\w+\s*\(|class\s+\w+\s*[\(:]|function\s+\w+\s*\(|for\s+\w+\s+in\s+|if\s+.*:|import\s+\w+|from\s+\w+\s+import|\w+\s*=\s*\w+\s*\(|console\.log\(|print\s*\(', text):
        return False
    
    copy_keywords = ['copy', 'repeat', 'reverse', 'constrained_response']
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


def check_constraint_following(response, ground_truth, extra_info, no_hacking):
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
        # constraint_data.append((extra_info['index'], f'constr_raw-{instr}', float(follow_instr), extra_info['split']))
        constraint_data.append((extra_info['index'], f'constr_nrh-{instr}', float(follow_instr and no_hacking), extra_info['split']))
    write_data(constraint_data)
    instr_level_reward = np.mean(constraint_eval.follow_instruction_list) #max_length_normalized(constraint_eval.follow_instruction_list, base=1.5)
    return instr_level_reward

def thinking_microsections(thinking):
    """Function to extract draft, analyze, and verify triples."""
    
    triples = []
    
    # Define regex patterns for each section
    draft_pattern = r'<<draft>>(.*?)<</draft>>'
    analyze_pattern = r'<<analyze>>(.*?)<</analyze>>'
    verify_pattern = r'<<verify>>(.*?)<</verify>>'
    
    # Find all matches for each section
    drafts = re.findall(draft_pattern, thinking, re.DOTALL)
    analyses = re.findall(analyze_pattern, thinking, re.DOTALL)
    verifications = re.findall(verify_pattern, thinking, re.DOTALL)
    
    # Combine into triples (assuming the correct order)
    format_reward = sum([len(drafts) > 0, len(analyses) > 0, len(verifications) > 0])/3
    min_len = max(len(drafts), len(verifications))
    
    for i in range(min_len):
        triple = (
            drafts[i].strip() if i < len(drafts) else None,
            analyses[i].strip() if i < len(analyses) else None,
            verifications[i].strip() if i < len(verifications) else None
        )
        triples.append(triple)
    
    return triples, format_reward

def check_constraint_accuracy(response, ground_truth, no_hacking):
    constraints = ground_truth["instruction_id"]
    eval_constraints = ground_truth['kwargs']
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
    if no_hacking:
        correctness = constraint_eval.follow_instruction_list
    else:
        correctness = [False for corr in constraint_eval.follow_instruction_list]
    auto_scores = dict(zip(constraint_eval.instruction_id_list, correctness))
    return auto_scores

def compute_verification_score(auto, judge):
    """
    Reward accurate self-assessment.
    Penalize overconfidence on errors more than underconfidence on correct answers.
    """
    if judge == 'nm':
        return 0.3  # Small default for parse failures
    elif judge == 'unsure':
        return 0.4  # Slight penalty vs confident correct
    elif auto and judge == 'true':
        return 1.0  # Correct and knows it
    elif not auto and judge == 'false':
        return 1.0  # Incorrect and knows it
    elif auto and judge == 'false':
        return 0.7  # Correct but cautious (not too bad!)
    else:  # auto == 0 and judge == 'true'
        return 0.0  # Hallucinating correctness (worst)

def agg_scores_per_constraint(scores_by_attempt):
    constraint_ids = scores_by_attempt[0].keys()
    total_reward = 0
    
    # Track metrics across all constraints
    all_cc_values = []
    all_jc_values = []
    all_cc_deltas = []  # last - first
    all_jc_deltas = []  # last - first
    
    for constraint_id in constraint_ids:
        remaining = 1
        constraint_reward = 0
        
        # Collect trajectory for this constraint
        cc_trajectory = []
        jc_trajectory = []
        
        for attempt_idx, attempt in enumerate(scores_by_attempt):
            cc, jc = attempt[constraint_id]
            cc_trajectory.append(cc)
            jc_trajectory.append(jc)
            
            term = remaining * cc * jc
            constraint_reward += term
            remaining *= (1 - cc * jc)
        
        total_reward += constraint_reward
        
        # Store trajectory data
        all_cc_values.extend(cc_trajectory)
        all_jc_values.extend(jc_trajectory)
        
        # Calculate deltas (last - first)
        if len(cc_trajectory) > 1:
            all_cc_deltas.append(cc_trajectory[-1] - cc_trajectory[0])
            all_jc_deltas.append(jc_trajectory[-1] - jc_trajectory[0])
    
    avg_reward = total_reward / len(constraint_ids)
    
    # Calculate statistics
    metrics = {
        'avg_reward': avg_reward,
        'cc_stats': {
            'mean': np.mean(all_cc_values),
            'std': np.std(all_cc_values),
        },
        'jc_stats': {
            'mean': np.mean(all_jc_values),
            'std': np.std(all_jc_values),
        },
        'cc_delta': {
            'mean': np.mean(all_cc_deltas) if all_cc_deltas else 0,
            'std': np.std(all_cc_deltas) if all_cc_deltas else 0,
        },
        'jc_delta': {
            'mean': np.mean(all_jc_deltas) if all_jc_deltas else 0,
            'std': np.std(all_jc_deltas) if all_jc_deltas else 0,
        }
    }
    
    return avg_reward, metrics

def draft_verification_match(draft, verify, ground_truth, no_hacking):
    """Scores a single draft, verify pair"""
    auto_scores = check_constraint_accuracy(draft, ground_truth, no_hacking) # score draft, will be dict with constraint labels
    print('auto_scores', auto_scores)
    verify_scores = extract_verify_logic(verify, ground_truth['instruction_id'])
    print('verify_scores', verify_scores)
    cc_jc = {}
    for constraint in ground_truth['instruction_id']:
        auto = auto_scores[constraint]
        judge = verify_scores.get(constraint, 'nm')
        cc_jc[constraint] = (int(auto), compute_verification_score(auto, judge))
    print('cc_jc', cc_jc)
    return cc_jc

def compute_score_single(solution_str, ground_truth, extra_info, data_source, diversity_score=0.0):
    """Score a single response with optional diversity bonus."""
    response = extract_xml_answer(solution_str, 'response')
    thinking = extract_xml_answer(solution_str, 'thinking')

    # Format rewards
    think_format, thoughts = follows_tag_format(solution_str, 'thinking')
    resp_format, responses = follows_tag_format(solution_str, 'response')
    triples, mt_format = thinking_microsections(thinking)
    format_reward = np.mean([think_format, resp_format, mt_format])
    
    # Prevent reward hacking
    min_unique_words = len(set(response.split(' '))) > 5
    not_fuzzy_pattern = not is_fuzzy_pattern(response, ground_truth["instruction_id"], threshold=0.7)
    not_constraint_in_resp = not constraint_in_response(response, extra_info['constraints'], ground_truth["instruction_id"])
    no_hacking = min_unique_words and not_fuzzy_pattern and not_constraint_in_resp
    
    format_data = [
        (extra_info['index'], 'format-think_format', float(think_format), extra_info['split']),
        (extra_info['index'], 'format-resp_format', float(resp_format), extra_info['split']),
        (extra_info['index'], 'format-micro_think_format', float(mt_format), extra_info['split']),
        (extra_info['index'], 'hack-min_unique_words', float(min_unique_words), extra_info['split']),
        (extra_info['index'], 'hack-not_fuzzy_pattern', float(not_fuzzy_pattern), extra_info['split']),
        (extra_info['index'], 'hack-not_constraint_in_resp', float(not_constraint_in_resp), extra_info['split']),
        (extra_info['index'], 'hack-no_hacking', float(no_hacking), extra_info['split']),
    ]
    write_data(format_data)
    
    # Constraint reward   
    constraint_reward = check_constraint_following(response, ground_truth, extra_info, no_hacking) 
    if not triples:
        mt_reward = 0
    else:
        n_constraints = len(ground_truth['instruction_id'])
        scores = []
        for triple in triples:
            cc_jc = draft_verification_match(triple[0], triple[2], ground_truth, no_hacking)
            # print(verify_score, '|', corr_score)
            # scores.append((corr_score, verify_score))
            scores.append(cc_jc)
        mt_pct_reward, metrics = agg_scores_per_constraint(scores)
        write_data_jsonl([metrics])
        mt_reward = n_constraints*mt_pct_reward
    

    # Calculate final reward
    if not no_hacking:
        final_reward = -1 # discourage hacking
        format_multiplier = 0
    elif constraint_reward == 0:
        format_multiplier = 0.5*format_reward # scale reward to 0 based on formatting [0, 0.5]
        final_reward = -0.5 + format_multiplier
    else:
        format_multiplier = 0.5 + 0.5*format_reward # scale reward based on formatting [0.5,1]
        final_reward = constraint_reward*mt_reward*format_multiplier
    reward_data = [
        (extra_info['index'], 'train-format_multiplier', float(format_multiplier), extra_info['split']),
        (extra_info['index'], 'train-constraint_reward', float(constraint_reward), extra_info['split']),
        (extra_info['index'], 'train-constraint_reward-nh', float(constraint_reward if no_hacking else 0), extra_info['split']),
        (extra_info['index'], 'train-mt_reward', float(mt_reward), extra_info['split']),
        (extra_info['index'], 'train-final_reward', float(final_reward), extra_info['split']),
    ]
    write_data(reward_data)
    
    do_print = random.randint(1, 128) == 1 # print avg 4 per step
    if do_print:        
        print(f"--------------------------------")
        print(f"final_reward: {final_reward}")
        print(f"constraint_reward: {constraint_reward} | mt_reward: {mt_reward} | format_reward: {format_reward} | format_multiplier: {format_multiplier}")
        print(f"think_format: {think_format} | resp_format: {resp_format} | mt_format: {mt_format}")
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
        thinking = [extract_xml_answer(sol, 'thinking') for sol in solution_str]
        
        # Compute diversity scores for all responses in this batch
        diversity_think = compute_diversity_scores(thinking, threshold=0.7)
        diversity_resp = compute_diversity_scores(responses, threshold=0.7)

        # Process each item in the batch with its diversity score
        scores = []
        for sol, gt, ei, ds, div_think, div_resp in zip(solution_str, ground_truth, extra_info, data_source, diversity_think, diversity_resp):
            score = compute_score_single(sol, gt, ei, ds, diversity_score=1.0)
            reward_data = [
                (ei['index'], 'train-diversity_think', float(div_think), ei['split']),
                (ei['index'], 'train-diversity_resp', float(div_resp), ei['split']),
            ]
            write_data(reward_data)
            scores.append(score)
        return scores
    else:
        # Single item processing - no diversity bonus
        print('ERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRROR should not be here')
        return compute_score_single(solution_str, ground_truth, extra_info, data_source, diversity_score=1.0)