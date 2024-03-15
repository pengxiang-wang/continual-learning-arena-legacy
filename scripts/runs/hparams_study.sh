#!/bin/bash
# Run from root folder with: bash scripts/runs/example.sh

# Schedule execution of many runs


# python src/train.py experiment=AdaHAT_1 trainer=gpu

# alpha=("1e-6 "2e-6" "3e-6" "4e-6" "5e-6" "6e-6" "7e-6" "8e-6" "9e-6" "1e-5")



# alpha=(1e-6 2e-6 3e-6 4e-6 5e-6 6e-6 7e-6 8e-6 9e-6 1e-5)
alpha=(1e-7 2e-7 3e-7 4e-7 5e-7 6e-7 7e-7 8e-7 9e-7)



for item in "${alpha[@]}"
do
    python src/train.py experiment=AdaHAT_1 trainer=gpu experiment_name=hparams_study model.adjust_strategy=ada model.alpha=$item
done