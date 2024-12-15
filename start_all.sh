# Get HF_TOKEN from local environment
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set"
    exit 1
fi

# Run test script on all TPU workers in parallel
for i in {0..7}; do
  echo "Starting worker $i..."
  gcloud compute tpus tpu-vm ssh finetune-70b \
    --zone=us-central2-b \
    --worker=$i -- "export HF_TOKEN=$HF_TOKEN; bash -l -s" < start_train.sh 2>&1 | tee test_worker_${i}.log &
done

# Wait for all workers to complete
wait
echo "All workers finished. Check test_worker_*.log files for results."
