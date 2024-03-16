#!/bin/bash
# Run from root folder with: bash scripts/runs/example.sh

# Schedule execution of many runs

for i in {5..10}
do
    python src/train.py trainer=gpu experiment=AdaHAT_1 experiment_name=ada_random model.adjust_strategy=ada_random seed=$i
done