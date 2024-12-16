#!/bin/bash

TPU_NODE="finetune-70b"
ZONE="us-central2-b"
NUM_WORKERS=8

for i in $(seq 0 $((NUM_WORKERS-1))); do
  echo "=== Cleaning checkpoints on worker $i ==="
  gcloud compute tpus tpu-vm ssh "$TPU_NODE" \
    --zone="$ZONE" \
    --worker="$i" -- '
    echo "Removing old checkpoints..."
    rm -rf /dev/shm/checkpoints/
    mkdir -p /dev/shm/checkpoints/
    echo "Checkpoint directory cleaned on worker $i"
    ' 2>&1 | tee cleanup_checkpoints_${i}.log &
done

wait

echo "Completed cleaning checkpoints on all workers"
