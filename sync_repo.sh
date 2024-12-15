#!/bin/bash

TPU_NODE="finetune-70b"
ZONE="us-central2-b"
NUM_WORKERS=8
REPO_URL="git@github.com:antra-tess/felafax_rl.git"
TARGET_DIR="felafax_distr"

for i in $(seq 0 $((NUM_WORKERS-1))); do
  echo "=== Syncing repository on worker $i ==="
  gcloud compute tpus tpu-vm ssh "$TPU_NODE" \
    --zone="$ZONE" \
    --worker="$i" -- "
    cd ~
    rm -rf $TARGET_DIR
    # Add GitHub to known hosts
    mkdir -p ~/.ssh
    ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null
    echo 'Cloning repository...'
    git clone $REPO_URL $TARGET_DIR
    " 2>&1 | tee sync_worker_${i}.log &
done

# Wait for all background processes to complete
wait

echo "Completed syncing repository on all workers"
