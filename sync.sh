cd ~/felafax_rl
tar --exclude .venv --exclude .git -czf felafax_distr.tar.gz .
gsutil cp felafax_distr.tar.gz gs://your-bucket/felafax_distr.tar.gz

TPU_NODE="finetune-70b"
ZONE="us-central2-b"
NUM_WORKERS=8
BUCKET="finetune70b"

for i in $(seq 0 $((NUM_WORKERS-1))); do
  gcloud compute tpus tpu-vm ssh "$TPU_NODE" \
    --zone="$ZONE" \
    --worker="$i" -- \
    "mkdir -p ~/felafax_distr && gsutil cp gs://$BUCKET/felafax_distr.tar.gz ~/felafax_distr/ && tar -xzf ~/felafax_distr/felafax_distr.tar.gz -C ~/felafax_distr && rm ~/felafax_distr/felafax_distr.tar.gz"
done

