# In the script (start_train.sh):
#!/bin/bash
cd ~/felafax_distr
source .venv/bin/activate
uv pip install libtpu-nightly==0.1.dev20240731 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
uv pip install jaxlib[tpu]==0.4.31 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html --no-deps
uv pip install jax==0.4.31 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html --no-deps
uv pip install -r requirements.txt
uv pip install -e .
export JAX_PROCESS_COUNT=8
# Extract the number at the end of the hostname for process index
export JAX_PROCESS_INDEX=$(hostname | grep -oP "\d+$")
#python trainers/llama3_70b_test/train.py
