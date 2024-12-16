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
    echo 'Writing/updating config file...'
    CONFIG_FILE=~/.config/gcsfuse/config.yaml
    if [ -f \"\$CONFIG_FILE\" ]; then
        if grep -q 'enable-parallel-downloads:' \"\$CONFIG_FILE\"; then
            sed -i 's/enable-parallel-downloads:.*/enable-parallel-downloads: true/' \"\$CONFIG_FILE\"
        else
            echo 'enable-parallel-downloads: true' >> \"\$CONFIG_FILE\"
        fi
    else
        echo 'enable-parallel-downloads: true' > \"\$CONFIG_FILE\"
    fi
    echo 'Config updated on worker $i'
    " 2>&1 | tee gcsfuse_config_${i}.log &
done

wait

echo "Completed updating gcsfuse config on all workers"
