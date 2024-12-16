#!/bin/bash

TPU_NODE="finetune-70b"
ZONE="us-central2-b"
NUM_WORKERS=8
GCS_PATH="/mnt/gcs-bucket/llama-70b-files/llama-70b-files"
LOCAL_PATH="/tmp/model-shards"

# Define shard ranges for each worker based on JAX sharding requirements
declare -A SHARD_RANGES=(
    ["0"]="1 4"    # Worker 0: shards 1-4
    ["1"]="4 8"    # Worker 1: shards 4-8
    ["2"]="8 12"   # Worker 2: shards 8-12
    ["3"]="12 15"  # Worker 3: shards 12-15
    ["4"]="16 19"  # Worker 4: shards 16-19
    ["5"]="19 23"  # Worker 5: shards 19-23
    ["6"]="23 27"  # Worker 6: shards 23-27
    ["7"]="27 30"  # Worker 7: shards 27-30
)

for i in $(seq 0 $((NUM_WORKERS-1))); do
  echo "=== Copying shards for worker $i ==="
  gcloud compute tpus tpu-vm ssh "$TPU_NODE" \
    --zone="$ZONE" \
    --worker="$i" -- "
    # Create local directory
    mkdir -p $LOCAL_PATH
    
    # Get shard range for this worker
    read start_shard end_shard <<< \"${SHARD_RANGES[$i]}\"
    
    # Copy each shard in the range
    for shard in \$(seq \$start_shard \$end_shard); do
        # Pad shard number with zeros
        padded_shard=\$(printf \"%05d\" \$shard)
        shard_file="model-\$padded_shard-of-00030.safetensors"
        if [ ! -f "$LOCAL_PATH/\$shard_file" ]; then
            echo \"Copying shard \$padded_shard...\"
            cp $GCS_PATH/\$shard_file $LOCAL_PATH/
        else
            echo \"Shard \$padded_shard already exists, skipping...\"
        fi
    done
    
    # Also copy necessary config files if they don't exist
    for config_file in config.json tokenizer.json tokenizer_config.json special_tokens_map.json generation_config.json; do
        if [ ! -f "$LOCAL_PATH/\$config_file" ]; then
            echo \"Copying \$config_file...\"
            cp $GCS_PATH/\$config_file $LOCAL_PATH/
        else
            echo \"\$config_file already exists, skipping...\"
        fi
    done
    
    echo \"Copied shards and config files for worker $i\"
    ls -lh $LOCAL_PATH/
    " 2>&1 | tee copy_shards_${i}.log &
done

wait

echo "Completed copying shards for all workers"
