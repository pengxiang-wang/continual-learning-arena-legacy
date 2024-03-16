#!/bin/bash
# Run from root folder with: bash scripts/runs/example.sh

# Schedule execution of many runs

for i in {7..10}
do
    python src/train.py trainer=gpu experiment=LwF_1 seed=$i
done