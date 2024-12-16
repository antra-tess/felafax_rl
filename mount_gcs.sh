#!/bin/bash

TPU_NODE="finetune-70b"
ZONE="us-central2-b"
NUM_WORKERS=8
BUCKET_NAME="finetune70b"
MOUNT_POINT="/mnt/gcs-bucket"

for i in $(seq 0 $((NUM_WORKERS-1))); do
  echo "=== Configuring worker $i ==="
  
  # This command sequence will:
  # 1. Unmount if mounted
  # 2. Ensure user_allow_other is set in fuse.conf
  # 3. Mount with --allow-other
  # 4. Set permissions

  gcloud compute tpus tpu-vm ssh "$TPU_NODE" \
    --zone="$ZONE" \
    --worker=$i -- "bash -l -c '
      # Step 1: Unmount if mounted
      if mountpoint -q $MOUNT_POINT; then
        echo \"Unmounting $MOUNT_POINT...\"
        sudo fusermount -u $MOUNT_POINT
      fi

      # Ensure the directory exists
      sudo mkdir -p $MOUNT_POINT

      # Step 2: Ensure user_allow_other in fuse.conf
      # Check if user_allow_other is already there
      if ! grep -q \"user_allow_other\" /etc/fuse.conf; then
        echo \"Adding user_allow_other to /etc/fuse.conf...\"
        echo user_allow_other | sudo tee -a /etc/fuse.conf
      fi

      # Step 3: Mount with --allow-other and config file
      echo \"Mounting GCS bucket $BUCKET_NAME to $MOUNT_POINT...\"
      sudo gcsfuse -o allow_other --file-mode=777 --dir-mode=777 --implicit-dirs --config-file ~/.config/gcsfuse/config.yaml $BUCKET_NAME $MOUNT_POINT

      # Step 4: Adjust permissions on the mount point if needed
      # allow-other does the main job, but we can set directory perms
      sudo chmod 777 $MOUNT_POINT

      echo \"Worker $i setup complete.\"
    '"
done
