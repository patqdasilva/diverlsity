import json
import argparse
from pathlib import Path
from datasets import Dataset, load_dataset, load_from_disk
import pandas as pd
import ast

class MakeParquet(object):
    def __init__(self, args):
        # Generate and Save
        self.save_dir=Path(args.local_save_dir)
        self.generate_parquet()

    def load_data(self):
        # Read samples
        # train_data = load_dataset("allenai/IF_multi_constraints_upto5", cache_dir="/fs/scratch/PAS2836/pqd/datasets")['train']
        # train_data = load_from_disk("/fs/ess/PAS2836/pqd/dc_if_ppl/data/allenai/IF-MC-len_1024-mc2")['train']
        train_data = Dataset.from_pandas(pd.read_parquet("/fs/ess/PAS2836/pqd/dc_if_ppl/data/allenai/IF_multi_constraints_upto5/train-00000-of-00001.parquet"))
        # train_data = load_from_disk("/fs/ess/PAS2836/pqd/dc_if_ppl/data/allenai/IF-MC-len_1024-olmo-oc_0-tt_0.5")['train']
        # train_data = load_from_disk("/fs/ess/PAS2836/pqd/dc_if_ppl/data/allenai/IF-MC-len_1024-mc2-qwen-tt_0.5")['train']
    
        test_data = Dataset.from_pandas(pd.read_json("/fs/ess/PAS2836/pqd/dc_if_ppl/IFBench/data/IFBench_test.jsonl", lines=True))

        return train_data, test_data
    
    def make_map_fn(self, split):
        raise NotImplementedError()
    
    def generate_parquet(self):
        train_data, test_data = self.load_data()
        train_dataset = train_data.map(function=self.make_map_fn('train'), with_indices=True, remove_columns=list(train_data.column_names))
        test_dataset = test_data.map(function=self.make_map_fn('test'), with_indices=True, remove_columns=list(test_data.column_names))

        # Write dataset
        # train_dataset.to_parquet(self.save_dir / 'train-olmo_oc-0-tt_0.5.parquet')
        # test_dataset.to_parquet(self.save_dir / 'test-olmo_oc-0-tt_0.5.parquet')
        # train_dataset.to_parquet(self.save_dir / 'train-qwen-dt_1.0.parquet')
        # test_dataset.to_parquet(self.save_dir / 'test-qwen-dt_1.0.parquet')
        # train_dataset.to_parquet(self.save_dir / 'train-mc2.parquet')
        # test_dataset.to_parquet(self.save_dir / 'test-mc2.parquet')
        train_dataset.to_parquet(self.save_dir / 'train.parquet')
        test_dataset.to_parquet(self.save_dir / 'test.parquet')
        print(f"Save in {self.save_dir}")
  
    def format_prompt(self, example, split):
        SYSTEM_PROMPT = "Your goal is to answer the prompt from the context section while adhering to all formatting constraints.\n\n<context>\n{prompt}\n</context>\n\nFirst, reason about the question and formatting constraints using these XML tags:\n<think>\n[your thoughts here]\n</think>\nThen, provide your final answer to the question, making sure to adhere to the formatting constraints, using these XML tags:\n<response>\n[your response here]\n</response>"
        USER_PROMPT = ""
        THINKING_PROMPT = ""
        if split == 'train':
            prompt = example['messages'][0]['content']
        elif split == 'test':
            prompt = example['prompt']
        formatted_prompt  = SYSTEM_PROMPT.format(prompt=prompt)# + USER_PROMPT + THINKING_PROMPT
        return formatted_prompt


class IFConstraints(MakeParquet):
    def __init__(self, args):
        super().__init__(args)

    def make_map_fn(self, split):
        def process_fn(example, idx):
            question = self.format_prompt(example, split)
            if split == 'train':
                gt = ast.literal_eval(example['ground_truth'])[0]
                constraints = example['constraint']
            elif split == 'test':
                gt = {
                    'instruction_id': example['instruction_id_list'],
                    'kwargs': example['kwargs'],
                }
                constraints = ''
            data = {
                "data_source": 'IF_multi_constraints_upto5',
                "uid": idx,
                "prompt": [
                    {"role": "user", "content": question}
                ],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": gt
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'constraints': constraints,
                }
            }
            return data
        return process_fn
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_save_dir", default="~/data/if_constraints", help="The save directory for the preprocessed dataset."
    )
    args = parser.parse_args()
    IFConstraints(args)