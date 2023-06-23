#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/runs/hat.sh

python src/train.py experiment=hat/hat_pmnist logger=many_loggers
python src/train.py experiment=hat/hat_pcifar10 logger=many_loggers
python src/train.py experiment=hat/hat_pcifar100 logger=many_loggers
python src/train.py experiment=hat/hat_smnist logger=many_loggers
python src/train.py experiment=hat/hat_scifar10 logger=many_loggers
python src/train.py experiment=hat/hat_scifar100 logger=many_loggers