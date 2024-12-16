#!/bin/bash

TPU_NODE="finetune-70b"
ZONE="us-central2-b"
NUM_WORKERS=8

for i in $(seq 0 $((NUM_WORKERS-1))); do
  echo "=== Updating gcsfuse config on worker $i ==="
  gcloud compute tpus tpu-vm ssh "$TPU_NODE" \
    --zone="$ZONE" \
    --worker="$i" -- '
    echo "Creating gcsfuse config directory..."
    mkdir -p ~/.config/gcsfuse
    echo "Writing/updating config file..."
    CONFIG_FILE=~/.config/gcsfuse/config.yaml
    if [ -f "$CONFIG_FILE" ]; then
        # Check if parallel downloads is already enabled
        if grep -q "enable-parallel-downloads:" "$CONFIG_FILE"; then
            # Update existing setting
            sed -i 's/enable-parallel-downloads:.*/enable-parallel-downloads: true/' "$CONFIG_FILE"
        else
            # Add setting if it doesn't exist
            echo "enable-parallel-downloads: true" >> "$CONFIG_FILE"
        fi
    else
        # Create new config file
        cat > "$CONFIG_FILE" << EOL
enable-parallel-downloads: true
EOL
    fi
    echo "Config updated on worker $i"
    ' 2>&1 | tee gcsfuse_config_${i}.log &
done

wait

echo "Completed updating gcsfuse config on all workers"
