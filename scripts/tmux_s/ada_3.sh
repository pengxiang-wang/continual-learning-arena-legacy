#!/bin/bash
# Run from root folder with: bash scripts/runs/example.sh

# Schedule execution of many runs

for i in {1..10}
do
    python src/train.py trainer=gpu experiment=AdaHAT_3 experiment_name=ada_3 model.adjust_strategy=ada seed=$i trainer.devices=[1]
done