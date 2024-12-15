#!/bin/bash

TPU_NODE="finetune-70b"
ZONE="us-central2-b"
NUM_WORKERS=8

for i in $(seq 0 $((NUM_WORKERS-1))); do
  echo "=== Cleaning worker $i ==="
  gcloud compute tpus tpu-vm ssh "$TPU_NODE" \
    --zone="$ZONE" \
    --worker="$i" -- '
    cd ~/felafax_distr
    # List all items except .venv, handling hidden files correctly
    shopt -s dotglob
    for item in *; do
      if [ "$item" != ".venv" ] && [ "$item" != "." ] && [ "$item" != ".." ]; then
        echo "Removing $item on worker '$i'"
        rm -rf "$item"
      fi
    done
    shopt -u dotglob
    ' 2>&1 | tee clean_worker_${i}.log &
done

# Wait for all background processes to complete
wait

echo "Completed cleaning all workers"
