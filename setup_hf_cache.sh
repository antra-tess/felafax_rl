#!/bin/bash

TPU_NODE="finetune-70b"
ZONE="us-central2-b"
NUM_WORKERS=8

for i in $(seq 0 $((NUM_WORKERS-1))); do
  echo "=== Setting up HuggingFace cache on worker $i ==="
  gcloud compute tpus tpu-vm ssh "$TPU_NODE" \
    --zone="$ZONE" \
    --worker="$i" -- '
    echo "Creating HuggingFace cache directory in /dev/shm..."
    mkdir -p /dev/shm/huggingface_cache
    chmod 777 /dev/shm/huggingface_cache
    rm -rf ~/.cache/huggingface
    ln -s /dev/shm/huggingface_cache ~/.cache/huggingface
    echo "HuggingFace cache setup complete on worker $i"
    ' 2>&1 | tee hf_cache_setup_${i}.log &
done

wait

echo "Completed setting up HuggingFace cache on all workers"
