#!/bin/bash
# Run from root folder with: bash scripts/runs/example.sh

# Schedule execution of many runs

for i in {1..10}
do
    python src/train.py trainer=gpu experiment=AdaHAT_2 experiment_name=ada_cons_1 model.adjust_strategy=ada_cons_1 seed=$i
done