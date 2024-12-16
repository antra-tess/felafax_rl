# Run test script on all TPU workers in parallel
for i in {0..7}; do
  echo "Starting worker $i..."
  gcloud compute tpus tpu-vm ssh finetune-70b \
    --zone=us-central2-b \
    --worker=$i -- 'bash -l -s' < start_test.sh 2>&1 | tee test_worker_${i}.log &
done

# Wait for all workers to complete
wait
echo "All workers finished. Check test_worker_*.log files for results."
