```
gcs/users/<user_id>/
├── metadata/                          # System metadata
│   ├── user_info.json                # User profile & settings
│   │   {
│   │     "email": "user@example.com",
│   │     "created_at": "2024-03-01T00:00:00Z",
│   │     "status": "active",
│   │     "name": "John Doe",
│   │     "company": "Example Corp"
│   │   }
│   ├── jobs.json                     # Jobs listing & status
│   │   {
│   │     "jobs": [{
│   │       "job_id": "job_123",
│   │       "model_id": "model_456",
│   │       "dataset_id": "dataset_789",
│   │       "status": "running",
│   │       "gpu_type": "A100",
│   │       "created_at": "...",
│   │       "updated_at": "..."
│   │     }]
│   │   }
│   ├── datasets.json                 # Datasets catalog
│   │   {
│   │     "datasets": [{
│   │       "dataset_id": "dataset_789",
│   │       "name": "training_data_v1",
│   │       "format": "jsonl",
│   │       "size": 1024576,
│   │       "samples": 10000,
│   │       "created_at": "...",
│   │       "path": "/data/dataset_789/"
│   │     }]
│   │   }
│   ├── models.json                   # Models catalog
│   │   {
│   │     "models": [{
│   │       "model_id": "model_456",
│   │       "job_id": "job_123",
│   │       "name": "custom_llama_v1",
│   │       "base_model": "llama3-8b",
│   │       "created_at": "...",
│   │       "path": "/models/model_456/"
│   │     }]
│   │   }
│   ├── billing.json                  # Billing records
│   │   {
│   │     "current_period": {
│   │       "start_date": "2024-03-01",
│   │       "end_date": "2024-03-31",
│   │       "total_cost": 156.78,
│   │       "usage_breakdown": {
│   │         "A100": {
│   │           "seconds": 3600,
│   │           "cost_per_hour": 10.00,
│   │           "total_cost": 10.00
│   │         }
│   │       },
│   │       "jobs": [{
│   │         "job_id": "job123",
│   │         "gpu_type": "A100",
│   │         "duration_seconds": 3600,
│   │         "cost": 10.00
│   │       }]
│   │     }
│   │   }
│   ├── gpu_pricing.json             # GPU pricing config
│   │   {
│   │     "pricing": {
│   │       "A100": {
│   │         "cost_per_hour": 10.00,
│   │         "min_billing_seconds": 60
│   │       }
│   │     }
│   │   }
│   └── indexes/                     # Quick lookup indexes
│       ├── jobs_by_status.json
│       ├── models_by_base.json
│       └── datasets_by_format.json
│
├── jobs/                            # Training jobs
│   └── <job_id>/
│       ├── config.yml              # User training config
│       │   {
│       │     "base_model": "llama3-8b",
│       │     "epochs": 3,
│       │     "batch_size": 32,
│       │     "learning_rate": 0.0001
│       │   }
│       ├── status.json            # Real-time status
│       │   {
│       │     "status": "running",
│       │     "progress": 45,
│       │     "current_epoch": 2,
│       │     "gpu_metrics": {
│       │       "gpu_type": "A100",
│       │       "utilization": 95,
│       │       "memory_used_gb": 35
│       │     }
│       │   }
│       ├── job.log               # Training logs
│       ├── internal_config.json  # Internal settings
│       ├── checkpoints/         # Model checkpoints
│       │   ├── epoch_1/
│       │   ├── epoch_2/
│       │   └── best/
│       └── events/             # Training metrics
│           └── tensorboard/
│
├── data/                      # Training datasets
│   └── <dataset_id>/
│       ├── raw/              # Original uploaded files
│       │   └── data.jsonl
│       ├── processed/        # Processed data
│       │   ├── train.bin
│       │   └── val.bin
│       └── metadata.json     # Dataset metadata
│           {
│             "samples": 10000,
│             "format": "jsonl",
│             "preprocessing": {
│               "tokenizer": "llama",
│               "max_length": 512
│             }
│           }
│
├── models/                   # Fine-tuned models
│   └── <model_id>/
│       ├── weights/         # Model weights
│       │   ├── pytorch_model.bin
│       │   └── config.json
│       ├── metadata.json    # Model metadata
│       │   {
│       │     "base_model": "llama3-8b",
│       │     "parameters": "8B",
│       │     "training_job": "job_123",
│       │     "metrics": {
│       │       "loss": 0.123,
│       │       "perplexity": 4.56
│       │     }
│       │   }
│       └── artifacts/      # Additional artifacts
│           ├── tokenizer/
│           └── examples/
│
└── billing/                # Billing details
    ├── invoices/          # Monthly invoices
    │   ├── 2024-03/
    │   └── 2024-02/
    └── usage_logs/        # Detailed usage logs
        └── 2024-03/
            └── job_123_usage.log
```