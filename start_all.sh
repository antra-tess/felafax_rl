for i in {0..7}; do
  gcloud compute tpus tpu-vm ssh finetune-70b \
    --zone=us-central2-b \
    --worker=$i -- 'bash -l -s' < start_train.sh 2>&1 | tee worker_${i}.log &
done

# Optional: wait for all background processes to complete
wait
