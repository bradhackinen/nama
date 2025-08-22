#!/bin/bash

for i in {1..40}
do
    echo ""
    echo "Running iteration number $i"
    conda activate nama
    python compile_gpt_reviewed_matches.py
    conda activate openai
    python review_new_pairs.py
done