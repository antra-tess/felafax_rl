#!/bin/bash
cd ~/felafax_distr

# Pull latest changes
echo "Pulling latest changes..."
git fetch
git reset --hard origin/main

# Activate environment and set variables
source .venv/bin/activate
export JAX_PROCESS_COUNT=8
export JAX_PROCESS_INDEX=$(hostname | grep -oP "\d+$")

# Set HuggingFace cache directory
export HF_HOME=/mnt/gcs-bucket/huggingface_cache

# Run the training script
echo "Starting training on worker ${JAX_PROCESS_INDEX}"
python trainers/llama3_70b_test/train.py
