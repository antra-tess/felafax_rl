# Felafax - LLM Fine-tuning Platform

A comprehensive platform for fine-tuning and managing large language models across any GPU like Google TPU, AWS Trainium.

## Features

- 🚀 Simple CLI interface for model fine-tuning
- 💾 Efficient storage management for models and datasets
- 📊 Usage-based billing with real-time tracking
- 🔄 Job management and monitoring
- 🔒 Secure authentication and user isolation
- 📈 Training metrics and progress tracking

## Quick Start

### Installation

```bash
pip install felafax-cli
```

### Basic Usage

1. Authenticate:
```bash
felafax-cli auth login --token <your-token>
```

2. Initialize configuration:
```bash
felafax-cli tune init-config
```

3. Upload training data:
```bash
felafax-cli files upload path/to/data.jsonl
```

4. Start fine-tuning:
```bash
felafax-cli tune start --model <model_name> --config config.yml --dataset <dataset_id>
```

## Available Commands

### Training Management
- `tune init-config` - Initialize a new config file
- `tune start` - Start a new fine-tuning job
- `tune list` - List all training jobs
- `tune status` - Check job status
- `tune stop` - Stop a running job

### File Management
- `files list` - List files in storage
- `files upload` - Upload training data
- `files delete` - Delete a file

### Model Operations
- `model list` - List available models
- `model download` - Download a model
- `model chat` - Interactive chat with a model

## Configuration

Example `config.yml`:
```yaml
hyperparameters:
  learning_rate: 1.0e-05
  batch_size: 32
  n_epochs: 4
  warmup_ratio: 0.0
lora:
  enabled: false
  r: 8
  alpha: 8
  dropout: 0.0
```

## Development

### Setup Development Environment

1. Clone the repository
2. Build the package:
```bash
python3 -m build
```

3. Create and activate virtual environment:
```bash
python3 -m venv env
source env/bin/activate  # Unix/MacOS
# or
env\Scripts\activate     # Windows
```

4. Install for development:
```bash
pip install -e .
```

### Project Structure

```
felafax/
├── src/
│ ├── cli/            # CLI implementation
│ ├── server/         # API server
│ └── core/           # Core functionality
├── storage/          # Storage management
└── docs/            # Documentation
```

## Architecture

### System Components

1. **API Endpoints** - RESTful API for all operations
2. **CLI Interface** - Command-line tool for user interaction
3. **Storage System** - Organized storage structure for all assets
4. **Training System** - Job management and execution
5. **Billing System** - Usage tracking and cost management

### Storage Structure

```
gcs/users/<user_id>/
├── metadata/         # System metadata
├── jobs/            # Training jobs
├── data/            # Training datasets
├── models/          # Fine-tuned models
└── billing/         # Billing information
```

## Support

For more information, visit our documentation at https://docs.felafax.ai


