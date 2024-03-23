#!/bin/bash
# Run from root folder with: bash scripts/runs/example.sh

# Schedule execution of many runs

for i in {1..10}
do
    python src/train.py experiment=Reference_4 seed=$i trainer=gpu trainer.devices=[1]
done