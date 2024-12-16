#!/bin/bash

TPU_NODE="finetune-70b"
ZONE="us-central2-b"
NUM_WORKERS=8
LOCAL_PATH="/tmp/model-shards"

# Define shard ranges for each worker
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
  echo "=== Cleaning extra shards for worker $i ==="
  gcloud compute tpus tpu-vm ssh "$TPU_NODE" \
    --zone="$ZONE" \
    --worker="$i" -- "
    cd $LOCAL_PATH
    
    # Get shard range for this worker
    read start_shard end_shard <<< \"${SHARD_RANGES[$i]}\"
    
    # Remove any shard files outside our range
    for shard_file in model-*.safetensors; do
        if [ -f \"\$shard_file\" ]; then
            shard_num=\$(echo \$shard_file | grep -o '[0-9]\\+' | head -1)
            if [ \$shard_num -lt \$start_shard ] || [ \$shard_num -gt \$end_shard ]; then
                echo \"Removing \$shard_file on worker $i (outside range \$start_shard-\$end_shard)\"
                rm \"\$shard_file\"
            fi
        fi
    done
    
    # List remaining files
    echo \"Remaining files on worker $i:\"
    ls -lh
    " 2>&1 | tee cleanup_shards_${i}.log &
done

wait

echo "Completed cleaning extra shards on all workers"
