# Development Guide

## Installation

### From Test PyPI
```bash
pipx install --pip-args='--extra-index-url https://pypi.org/simple' -i https://test.pypi.org/simple/ felafax-cli==0.1.1
```

### For Development

1. Build the package:
```bash
python3 -m build
```

2. Test the installation:
```bash
# Create a test directory and virtual environment
mkdir -p ./tmp/
python3 -m venv env

# Activate the virtual environment
# On Unix/MacOS:
source env/bin/activate
# On Windows:
# env\Scripts\activate

# Install the package
pip install ../dist/felafax_cli-0.1.0.tar.gz

# Verify installation
felafax-cli --help
```

## Publishing

### To Test PyPI
```bash
python3 -m twine upload --repository testpypi dist/* --verbose
```

### To PyPI
```bash
python3 -m twine upload dist/* --verbose
```

Note: Make sure you have `twine` installed (`pip install twine`) and have configured your PyPI credentials before publishing.

## Development Workflow

1. Start the development server:
```bash
./scripts/dev_start_server.sh
```

2. Run the CLI in development mode:
```bash
python -m felafax.cli.main --help
```
