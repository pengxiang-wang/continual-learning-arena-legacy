#!/bin/bash
# Run from root folder with: bash scripts/runs/example.sh

# Schedule execution of many runs

for i in {1..10}
do
<<<<<<< HEAD
    python src/train.py trainer=gpu experiment=AdaHAT_2 experiment_name=ada_2 model.adjust_strategy=ada seed=$i
=======
    python src/train.py trainer=gpu experiment=AdaHAT_3 experiment_name=ada_3 model.adjust_strategy=ada seed=$i trainer.devices=[1]
>>>>>>> 9c7afd6c13702bfee76e43b16ea2dcbe75aacf5a
done