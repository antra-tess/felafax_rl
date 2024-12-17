#!/bin/bash

TPU_NODE="finetune-70b"
ZONE="us-central2-b"
NUM_WORKERS=8

for i in $(seq 0 $((NUM_WORKERS-1))); do
  echo "=== Updating config on worker $i ==="
  gcloud compute tpus tpu-vm ssh "$TPU_NODE" \
    --zone="$ZONE" \
    --worker="$i" -- '
    CONFIG_FILE="/tmp/model-shards/config.json"
    echo "Updating config file..."
    # Use jq to add the type field while preserving the rest of the config
    jq ".rope_scaling.type = .rope_scaling.rope_type" "$CONFIG_FILE" > "$CONFIG_FILE.tmp" && mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"
    echo "Config updated on worker $i"
    ' 2>&1 | tee update_config_${i}.log &
done

wait

echo "Completed updating config on all workers"
