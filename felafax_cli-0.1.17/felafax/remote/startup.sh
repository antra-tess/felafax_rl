#!/bin/bash
set -eo pipefail

if [ "$TORCH_XLA" = "1" ]; then
  # install pytorch stuff
  pip install torch~=2.3.0 torch_xla[tpu]~=2.3.0 torchvision -f https://storage.googleapis.com/libtpu-releases/index.html
  pip install --upgrade transformers
fi

echo 'export PJRT_DEVICE=TPU' >>~/.bashrc

# Load scripts
mkdir -p "/home/scripts/"
gsutil -m cp -r "gs://felafax-storage-v2/scripts/*" "/home/scripts/"

# Install script requirements
pip install -r "/home/scripts/requirements.txt"

# Execute the script 
if [ -n "$SCRIPT_NAME" ]; then
  python "/home/scripts/$SCRIPT_NAME"
else
  echo "No script to execute"
fi


# Start Jupyter Lab
# exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''