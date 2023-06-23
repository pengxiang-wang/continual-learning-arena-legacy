#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/runs/hat.sh

python src/train.py experiment=finetuning/finetuning_pmnist logger=many_loggers
python src/train.py experiment=finetuning/finetuning_cil_smnist logger=many_loggers
python src/train.py experiment=finetuning/finetuning_pcifar10 logger=many_loggers
python src/train.py experiment=finetuning/finetuning_pcifar100 logger=many_loggers
python src/train.py experiment=finetuning/finetuning_smnist logger=many_loggers
python src/train.py experiment=finetuning/finetuning_scifar10 logger=many_loggers
python src/train.py experiment=finetuning/finetuning_scifar100 logger=many_loggers
