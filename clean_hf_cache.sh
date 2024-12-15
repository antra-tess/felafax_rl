#!/bin/bash

TPU_NODE="finetune-70b"
ZONE="us-central2-b"
NUM_WORKERS=8

for i in $(seq 0 $((NUM_WORKERS-1))); do
  echo "=== Cleaning HuggingFace cache on worker $i ==="
  gcloud compute tpus tpu-vm ssh "$TPU_NODE" \
    --zone="$ZONE" \
    --worker="$i" -- '
    echo "Cleaning HuggingFace cache..."
    rm -rf ~/.cache/huggingface/
    echo "Cache cleaned on worker '$i'"
    ' 2>&1 | tee hf_cache_clean_${i}.log &
done

# Wait for all background processes to complete
wait

echo "Completed cleaning HuggingFace cache on all workers"
