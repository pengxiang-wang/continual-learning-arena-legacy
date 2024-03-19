#!/bin/bash
# Run from root folder with: bash scripts/runs/example.sh

# Schedule execution of many runs

for i in {1..10}
do
    python src/train.py trainer=gpu experiment=AdaHAT_3 experiment_name=ada_cons_1_3 model.adjust_strategy=ada_cons_1 seed=$i trainer.devices=[1]
done