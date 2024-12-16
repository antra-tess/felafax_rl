#!/bin/bash

TPU_NODE="finetune-70b"
ZONE="us-central2-b"
NUM_WORKERS=8
GCS_PATH="/mnt/gcs-bucket/llama-70b-files/llama-70b-files"
LOCAL_PATH="/tmp/model-shards"

# Define shard ranges for each worker based on JAX sharding requirements
declare -A SHARD_RANGES=(
    ["0"]="1 4"    # Worker 0: shards 1-4 (includes overlap)
    ["1"]="4 8"    # Worker 1: shards 4-8 (includes overlaps)
    ["2"]="8 12"   # Worker 2: shards 8-12 (includes overlaps)
    ["3"]="12 15"  # Worker 3: shards 12-15 (includes overlap)
    ["4"]="15 19"  # Worker 4: shards 15-19 (includes overlap)
    ["5"]="19 23"  # Worker 5: shards 19-23 (includes overlap)
    ["6"]="23 27"  # Worker 6: shards 23-27 (includes overlap)
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
        echo \"Copying shard \$padded_shard...\"
        cp $GCS_PATH/model-\$padded_shard-of-00030.safetensors $LOCAL_PATH/
    done
    
    # Also copy necessary config files
    cp $GCS_PATH/config.json $LOCAL_PATH/
    cp $GCS_PATH/tokenizer.json $LOCAL_PATH/
    cp $GCS_PATH/tokenizer_config.json $LOCAL_PATH/
    cp $GCS_PATH/special_tokens_map.json $LOCAL_PATH/
    cp $GCS_PATH/generation_config.json $LOCAL_PATH/
    
    echo \"Copied shards and config files for worker $i\"
    ls -lh $LOCAL_PATH/
    " 2>&1 | tee copy_shards_${i}.log &
done

wait

echo "Completed copying shards for all workers"
