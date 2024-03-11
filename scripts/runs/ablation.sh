#!/bin/bash
# Run from root folder with: bash scripts/runs/example.sh

# Schedule execution of many runs


# python src/train.py experiment=AdaHAT_1 trainer=gpu
python src/train.py experiment=AdaHAT_1 trainer=gpu model.adjust_strategy=ada_no_sum
python src/train.py experiment=AdaHAT_1 trainer=gpu model.adjust_strategy=ada_no_reg
python src/train.py experiment=AdaHAT_1 trainer=gpu model.adjust_strategy=random
python src/train.py experiment=AdaHAT_1 trainer=gpu model.adjust_strategy=constant