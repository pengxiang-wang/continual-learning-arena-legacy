#!/bin/bash
# Run from root folder with: bash scripts/runs/example.sh

# Schedule execution of many runs

for i in {5..5}
do
    python src/train.py trainer=gpu experiment=AdaHAT_2 experiment_name=ada_cons_alpha_2 model.adjust_strategy=ada_cons_alpha seed=$i
done