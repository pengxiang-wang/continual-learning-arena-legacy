d#!/bin/bash
# Run from root folder with: bash scripts/runs/example.sh

# Schedule execution of many runs


# python src/train.py experiment=AdaHAT_1 trainer=gpu

# alpha=("1e-6 "2e-6" "3e-6" "4e-6" "5e-6" "6e-6" "7e-6" "8e-6" "9e-6" "1e-5")



# alpha=(1e-6 2e-6 3e-6 4e-6 5e-6 6e-6 7e-6 8e-6 9e-6 1e-5)
# alpha=(1e-7 2e-7 3e-7 4e-7 5e-7 6e-7 7e-7 8e-7 9e-7)
alpha=(1e-7 5e-7 9e-7 1e-6 2e-6 5e-6 1e-5)
num_tasks=20
perm_seeds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]


for i in {1..10}
do
    for item in "${alpha[@]}"
    do
        python src/train.py experiment=AdaHAT_1 trainer=gpu experiment_name=hparams_study_$item model.adjust_strategy=ada model.alpha=$item data.num_tasks=$num_tasks data.perm_seeds=$perm_seeds seed=$i
    done
done