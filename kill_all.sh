#!/bin/bash

TPU_NODE="finetune-70b"
ZONE="us-central2-b"
NUM_WORKERS=8

for i in $(seq 0 $((NUM_WORKERS-1))); do
  echo "=== Starting process for worker $i ==="
  gcloud compute tpus tpu-vm ssh "$TPU_NODE" \
    --zone="$ZONE" \
    --worker="$i" -- '
    echo "Looking for processes using TPU on worker '$i'..."
    TPU_PROCS=$(sudo lsof -w /dev/accel0)
    if [ -z "$TPU_PROCS" ]; then
      echo "No processes found using TPU on worker '$i'"
    else
      echo "Found processes on worker '$i':"
      echo "$TPU_PROCS"
      PID=$(echo "$TPU_PROCS" | grep python | head -1 | awk "{print \$2}")
      if [ ! -z "$PID" ]; then
        echo "Killing PID: $PID on worker '$i'"
        sudo kill -9 $PID
        echo "Process killed on worker '$i'"
      else
        echo "Could not extract PID on worker '$i'"
      fi
    fi

    # Remove the TPU lockfile if it exists
    if [ -f /tmp/libtpu_lockfile ]; then
      echo "Removing lockfile on worker '$i'"
      sudo rm -f /tmp/libtpu_lockfile
    fi
    ' 2>&1 | tee worker_${i}_kill.log &
done

# Wait for all background processes to complete
wait

echo "Completed processing all workers"
