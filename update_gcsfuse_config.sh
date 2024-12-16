#!/bin/bash

TPU_NODE="finetune-70b"
ZONE="us-central2-b"
NUM_WORKERS=8

for i in $(seq 0 $((NUM_WORKERS-1))); do
  echo "=== Updating gcsfuse config on worker $i ==="
  gcloud compute tpus tpu-vm ssh "$TPU_NODE" \
    --zone="$ZONE" \
    --worker="$i" -- "
    echo 'Creating gcsfuse config directory...'
    mkdir -p ~/.config/gcsfuse
    echo 'Writing config file...'
    cat > ~/.config/gcsfuse/config.yaml << 'EOL'
file-cache:
  enable-parallel-downloads: true
EOL
    echo 'Config updated on worker $i'
    " 2>&1 | tee gcsfuse_config_${i}.log &
done

wait

echo "Completed updating gcsfuse config on all workers"
