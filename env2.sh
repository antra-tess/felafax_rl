mkdir ~/felafax_distr
cd ~/felafax_distr
uv venv --python 3.11.11
source .venv/bin/activate
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
