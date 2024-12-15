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
    # List all items except .venv
    items=$(ls -A | grep -v "^\.venv$")
    if [ -z "$items" ]; then
      echo "No items to remove on worker '$i'"
    else
      echo "Removing items on worker '$i':"
      echo "$items"
      for item in $items; do
        rm -rf "$item"
        echo "Removed $item"
      done
    fi
    ' 2>&1 | tee clean_worker_${i}.log &
done

# Wait for all background processes to complete
wait

echo "Completed cleaning all workers"
