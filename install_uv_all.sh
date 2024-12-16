for i in {0..7}; do
  gcloud compute tpus tpu-vm ssh finetune-70b \
    --zone=us-central2-b \
    --worker=$i -- 'bash -s' < install_uv.sh
done

