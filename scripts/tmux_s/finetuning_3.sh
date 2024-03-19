#!/bin/bash
# Run from root folder with: bash scripts/runs/example.sh

# Schedule execution of many runs

for i in {1..10}
do
    python src/train.py experiment=Finetuning_3 trainer=gpu trainer.devices=[1] seed=$i
done