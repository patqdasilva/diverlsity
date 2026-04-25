#!/bin/bash

echo `date`
source ~/miniconda3/bin/activate ctrlf

cd /fs/ess/PAS2836/pqd/dc_if_ppl/diverlsity/examples/data_preprocess/instruction_following
python -m if_constraints_parquet --local_save_dir '/fs/ess/PAS2836/pqd/dc_if_ppl/diverlsity/examples/data_preprocess/instruction_following/IF-RLVR-mc'