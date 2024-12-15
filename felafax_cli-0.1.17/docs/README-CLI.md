---
title: 'CLI Reference'
description: 'Command-line interface for fine-tuning and managing large language models'
---

## Installation

Install the Felafax CLI using pip:

```bash
pip install felafax-cli
```

## Authentication

Before using the CLI, authenticate with your token:

```bash
felafax-cli auth login --token <your-token>
```

<Tip>
Use the `--force` flag to override existing authentication.
</Tip>

## Core Features

<CardGroup cols={2}>
  <Card
    title="Model Management"
    icon="robot"
    href="#model-management"
  >
    List, chat with, and manage your fine-tuned models
  </Card>
  <Card
    title="Training Management"
    icon="graduation-cap"
    href="#training-management"
  >
    Initialize and manage fine-tuning jobs
  </Card>
  <Card
    title="File Management"
    icon="folder"
    href="#file-management"
  >
    Upload and manage your training data files
  </Card>
  <Card
    title="Configuration"
    icon="gear"
    href="#configuration"
  >
    Configure your fine-tuning parameters
  </Card>
</CardGroup>

## Model Management

### List Models

View all your available fine-tuned models:

```bash
felafax-cli model list
```

### Interactive Chat

Start an interactive chat session with a model:

```bash
felafax-cli model chat <model_id> [--system-prompt "Your custom prompt"]
```

### Model Information

Get detailed information about a specific model:

```bash
felafax-cli model info <model_id>
```

### Delete Model

Remove a fine-tuned model:

```bash
felafax-cli model delete <model_id>
```

## Training Management

### Initialize Configuration

Create a new configuration file:

```bash
felafax-cli tune init-config
```

### Start Fine-tuning

Launch a new fine-tuning job:

```bash
felafax-cli tune start \
    --model <model_name> \
    --config path/to/config.yml \
    --dataset <dataset_id>
```

### Manage Jobs

```bash
# List all training jobs
felafax-cli tune list

# Check job status
felafax-cli tune status --job-id <id>

# Stop a running job
felafax-cli tune stop --job-id <id>
```

## File Management

### List Files

View all your training data files:

```bash
felafax-cli files list [--prefix <prefix>] [--limit <number>]
```

### Upload File

Upload a new training data file:

```bash
felafax-cli files upload <file_path>
```

### Delete File

Remove a training data file:

```bash
felafax-cli files delete <file_path>
```

## Configuration

### Configuration Example

When you run `felafax-cli tune init-config`, it creates a YAML file with the following structure:

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

## Support

For more information, visit our documentation at <https://docs.felafax.ai>

## License

MIT License

