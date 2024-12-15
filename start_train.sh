#!/bin/bash
cd ~/felafax_distr
source .venv/bin/activate

# Set JAX process coordination variables
export JAX_PROCESS_COUNT=8
export JAX_PROCESS_INDEX=$(hostname | grep -oP "\d+$")

# Run the training script
echo "Starting training on worker ${JAX_PROCESS_INDEX}"
python trainers/llama3_70b_test/train.py
