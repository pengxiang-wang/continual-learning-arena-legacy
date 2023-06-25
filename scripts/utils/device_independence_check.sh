#!/bin/bash
# Run from root folder with: bash scripts/runs/*.sh

# Example experiments with different strategies (cpu, gpu, ddp, etc), to make sure they produce similar results
# Check the test_metrics.csv file

python src/train.py trainer=cpu logger=csv experiment_name=cpu
python src/train.py trainer=gpu logger=csv experiment_name=gpu_single
python src/train.py trainer=ddp trainer.num_nodes=1 trainer.devices=8 experiment_name=ddp_multi_gpus
python src/train.py trainer=ddp trainer.num_nodes=2 trainer.devices=4 experiment_name=ddp_multi_gpus_nodes