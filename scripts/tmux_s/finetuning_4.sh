#!/bin/bash
# Run from root folder with: bash scripts/runs/example.sh

# Schedule execution of many runs

for i in {5..10}
do
    python src/train.py experiment=Finetuning_4 trainer=gpu trainer.devices=[1] seed=$i
done