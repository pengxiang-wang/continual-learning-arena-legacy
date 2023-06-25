#!/bin/bash
# check sanity of a shell script in scripts/runs by running 1 epoch 
# Run from root folder with: bash scripts/utils/cleanup.sh

# loop through lines of .sh (if starts with python), add str to last, execute

trainer.max_epochs=1

debug?