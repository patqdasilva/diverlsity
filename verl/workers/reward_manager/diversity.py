# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("diversity")
class DiversityRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the DiversityRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def _call_grouped_by_uid(self, data: DataProto, return_dict=False):
        """Group responses by UID and score together."""
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}
        
        # Group indices by UID
        uid_groups = defaultdict(list)
        uids = data.non_tensor_batch["uid"]
        
        for i in range(len(data)):
            uid = uids[i]
            uid_groups[uid].append(i)
        
        # Process each UID group
        for uid, indices in uid_groups.items():
            # Collect all data for this UID
            responses_str = []
            ground_truths = []
            data_sources = []
            extra_infos = []
            valid_response_lengths = []
            prompt_strs = []
            
            for i in indices:
                data_item = data[i]
                
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                
                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]
                
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                
                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"] if "reward_model" in data_item.non_tensor_batch and "ground_truth" in data_item.non_tensor_batch["reward_model"] else None
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                extra_info = data_item.non_tensor_batch.get("extra_info", None)
                
                responses_str.append(response_str)
                ground_truths.append(ground_truth)
                data_sources.append(data_source)
                extra_infos.append(extra_info)
                valid_response_lengths.append(valid_response_length)
                prompt_strs.append(prompt_str)
            
            # Call compute_score with lists (one call per UID group)
            scores = self.compute_score(
                data_source=data_sources,
                solution_str=responses_str,
                ground_truth=ground_truths,
                extra_info=extra_infos,
            )
            
            # Assign rewards
            for idx, (i, valid_length) in enumerate(zip(indices, valid_response_lengths)):
                score = scores[idx]
                
                if isinstance(score, dict):
                    reward = score["score"]
                    for key, value in score.items():
                        reward_extra_info[key].append(value)
                else:
                    reward = score
                
                reward_tensor[i, valid_length - 1] = reward
                
                # Print for examination
                data_source = data_sources[idx]
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0
                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print(f"[uid={uid}, response {idx+1}/{len(indices)}]")
                    print("[prompt]", prompt_strs[idx])
                    print("[response]", responses_str[idx])
                    print("[ground_truth]", ground_truths[idx])
                    if isinstance(score, dict):
                        for key, value in score.items():
                            print(f"[{key}]", value)
                    else:
                        print("[score]", score)
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
