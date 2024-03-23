#!/bin/bash
# Run from root folder with: bash scripts/runs/example.sh

# Schedule execution of many runs

for i in {4..10}
do
    python src/train.py experiment=Random_2 seed=$i
done