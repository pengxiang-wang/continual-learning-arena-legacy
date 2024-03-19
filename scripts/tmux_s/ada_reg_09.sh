#!/bin/bash
# Run from root folder with: bash scripts/runs/example.sh

# Schedule execution of many runs

for i in {1..10}
do
    python src/train.py trainer=gpu experiment=AdaHAT_1 experiment_name=ada_reg_09 model.adjust_strategy=ada_reg_09 seed=$i
done