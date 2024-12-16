#!/bin/bash

TPU_NODE="finetune-70b"
ZONE="us-central2-b"
NUM_WORKERS=8

for i in $(seq 0 $((NUM_WORKERS-1))); do
  echo "=== Removing HuggingFace symlink on worker $i ==="
  gcloud compute tpus tpu-vm ssh "$TPU_NODE" \
    --zone="$ZONE" \
    --worker="$i" -- '
    echo "Removing HuggingFace symlink..."
    rm -f ~/.cache/huggingface
    echo "Symlink removed on worker $i"
    ' 2>&1 | tee remove_symlink_${i}.log &
done

wait

echo "Completed removing HuggingFace symlinks on all workers"
