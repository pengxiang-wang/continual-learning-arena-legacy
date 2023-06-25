#!/bin/bash
# Run from root folder with: bash scripts/runs/example.sh

# Schedule execution of many runs


python src/train.py trainer.max_epochs=5 logger=csv

python src/train.py trainer.max_epochs=10 logger=csv
