#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/runs/hat.sh

# basic TIL scenario
python src/train.py experiment=finetuning_mlp/pmnist logger=many_loggers
# basic CIL scenario
python src/train.py experiment=finetuning_mlp/smnist_cil logger=many_loggers

python src/train.py experiment=finetuning_mlp/pcifar10 logger=many_loggers
python src/train.py experiment=finetuning_mlp/pcifar100 logger=many_loggers
python src/train.py experiment=finetuning_mlp/pomniglot logger=many_loggers

python src/train.py experiment=finetuning_mlp/smnist logger=many_loggers
python src/train.py experiment=finetuning_mlp/scifar10 logger=many_loggers
python src/train.py experiment=finetuning_mlp/scifar100 logger=many_loggers
python src/train.py experiment=finetuning_mlp/somniglot logger=many_loggers