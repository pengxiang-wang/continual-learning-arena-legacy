#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/runs/hat.sh

#python src/train.py experiment=hat_mlp/pmnist logger=many_loggers
python src/train.py experiment=hat_mlp/pcifar10 logger=many_loggers
python src/train.py experiment=hat_mlp/pcifar100 logger=many_loggers

# python src/train.py experiment=hat_mlp/smnist logger=many_loggers
# python src/train.py experiment=hat_mlp/scifar10 logger=many_loggers
# python src/train.py experiment=hat_mlp/scifar100 logger=many_loggers