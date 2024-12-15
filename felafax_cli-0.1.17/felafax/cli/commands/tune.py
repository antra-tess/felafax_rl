import typer
from datetime import datetime
import yaml
from ..common import get_user_id, require_auth, get_server_uri, log_event
import httpx
from pathlib import Path
from ...core.constants import SUPPORTED_MODELS
import sys
from time import sleep

tune_app = typer.Typer(help="Tuning commands")

@tune_app.command("init-config")
@require_auth
def init_config():
    """Initialize a new fine-tuning configuration file"""
    
    config = {
        'data_config': {
            'batch_size': 16,
            'max_seq_length': 2048,
            'dataset_input_field': 'instruction',
            'dataset_output_field': 'output'
        },
        'trainer_config': {
            'param_dtype': 'bfloat16',
            'compute_dtype': 'bfloat16', 
            'num_epochs': 1,
            'num_steps': 5,
            'learning_rate': 1e-3,
            'lora_rank': 16,
            'use_lora': True,
            'log_interval': 5,
            'eval_interval': 5,
            'eval_steps': 10
        },
        'huggingface_config': {
            'hf_repo': '',  
            'hf_token': ''  
        }
    }
    
    # Generate filename with date prefix
    date_suffix = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    filename = f"felafax-finetune-{date_suffix}.yml"
    
    with open(filename, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    typer.echo(f"Created fine-tuning configuration file: {filename}") 

@tune_app.command("start")
@require_auth
def start_tuning(
    model: str = typer.Option(..., "--model", "-m", help=f"Base model to fine-tune (one of: {', '.join(SUPPORTED_MODELS)})"),
    config: str = typer.Option(..., "--config", "-c", help="Path to config YAML file"), 
    dataset_id: str = typer.Option(None, "--dataset-id", "-d", help="Dataset ID to use for training"),
    hf_dataset_path: str = typer.Option(None, "--hf-dataset", "-h", help="HuggingFace dataset path to use for training")
):
    """Start a new fine-tuning job"""
    # Validate model and dataset parameters
    if model not in SUPPORTED_MODELS:
        log_event("tuning_start_failed", {"reason": "invalid_model", "model": model})
        typer.echo(f"Error: Invalid model. Must be one of: {', '.join(SUPPORTED_MODELS)}")
        raise typer.Exit(1)
    
    if not dataset_id and not hf_dataset_path:
        log_event("tuning_start_failed", {"reason": "missing_dataset"})
        typer.echo("Error: Either dataset-id or hf-dataset must be provided")
        raise typer.Exit(1)
    
    try:
        # Load and validate config file
        config_path = Path(config)
        if not config_path.exists():
            typer.echo(f"Error: Config file not found: {config}")
            raise typer.Exit(1)
            
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        # Prepare request payload
        request_data = {
            "model_name": model,
            "config": config_data
        }
        
        # Add either dataset_id or hf_dataset_path to the request
        if dataset_id:
            request_data["dataset_id"] = dataset_id
        if hf_dataset_path:
            request_data["hf_dataset_path"] = hf_dataset_path

        user_id = get_user_id()
        
        # Make API request
        try:
            server_uri = get_server_uri()
            response = httpx.post(
                f"{server_uri}/fine-tune/{user_id}/start",
                json=request_data,
                timeout=300  # 5 minutes in seconds
            )
            
            if response.status_code == 200:
                result = response.json()
                log_event("tuning_started", {
                    "model": model,
                    "tune_id": result['tune_id'],
                    "dataset_id": dataset_id
                })
                typer.echo(f"Started fine-tuning job: {result['tune_id']}")
                typer.echo(f"Status: {result['status']}")
                typer.echo(f"Message: {result['message']}")
            else:
                log_event("tuning_start_failed", {
                    "reason": "api_error",
                    "status_code": response.status_code
                })
                typer.echo(f"Error: {response.status_code} - {response.text}")
                raise typer.Exit(1)
                
        except httpx.RequestError as e:
            typer.echo(f"Error connecting to API: {e}")
            raise typer.Exit(1)
    except Exception as e:
        log_event("tuning_start_failed", {
            "reason": "exception",
            "error_type": type(e).__name__
        })
        typer.echo(f"Error: {str(e)}")
        raise typer.Exit(1)

@tune_app.command("list")
def list_jobs():
    """List all fine-tuning jobs"""
    log_event("tuning_list_viewed")
    user_id = get_user_id()
    server_uri = get_server_uri()
    
    try:
        response = httpx.get(f"{server_uri}/fine-tune/{user_id}/list")
        response.raise_for_status()
        jobs = response.json()
        
        if not jobs:
            typer.echo("No fine-tuning jobs found")
            return
            
        # Define column widths
        col_widths = {
            'tune_id': 35,
            'base_model': 20,
            'status': 12,
            'created_at': 10
        }
        
        # Print header
        typer.echo("\nFine-tuning Jobs:")
        typer.echo("-" * (sum(col_widths.values()) + 3 * len(col_widths) - 1))
        
        # Print column headers
        header = (
            f"{'Job ID':<{col_widths['tune_id']}} | "
            f"{'Base Model':<{col_widths['base_model']}} | "
            f"{'Status':<{col_widths['status']}} | "
            f"{'Created At':<{col_widths['created_at']}}"
        )
        typer.echo(header)
        typer.echo("-" * (sum(col_widths.values()) + 3 * len(col_widths) - 1))
        
        # Print each job
        for job in jobs:
            created_at = job['created_at'].split('T')[0]  # Just get the date part
            row = (
                f"{job['tune_id']:<{col_widths['tune_id']}} | "
                f"{job['base_model']:<{col_widths['base_model']}} | "
                f"{job['status']:<{col_widths['status']}} | "
                f"{created_at:<{col_widths['created_at']}}"
            )
            typer.echo(row)
            
    except httpx.RequestError as e:
        typer.echo(f"Error connecting to API: {e}")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: {str(e)}")
        raise typer.Exit(1)

@tune_app.command("status")
@require_auth
def get_status(
    job_id: str = typer.Option(..., help="Job ID to check status for")
):
    """Get status of a specific fine-tuning job"""
    log_event("tuning_status_checked", {"tune_id": job_id})
    user_id = get_user_id()
    server_uri = get_server_uri()
    
    try:
        response = httpx.get(f"{server_uri}/fine-tune/{user_id}/{job_id}/status")
        response.raise_for_status()
        status = response.json()
        
        # Print status information
        typer.echo("\nFine-tuning Job Status:")
        typer.echo("-" * 40)
        typer.echo(f"Job ID: {status['tune_id']}")
        typer.echo(f"Status: {status['status']}")
        typer.echo(f"Created: {status['created_at']}")
        typer.echo(f"Last Updated: {status['updated_at']}")
        if status.get('progress') is not None:
            typer.echo(f"Progress: {status['progress']:.1%}")
            
    except httpx.RequestError as e:
        typer.echo(f"Error connecting to API: {e}")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: {str(e)}")
        raise typer.Exit(1)

@tune_app.command("stop")
@require_auth
def stop_job(
    job_id: str = typer.Option(..., "-j", "--job-id", help="Job ID to stop")
):
    """Stop a running fine-tuning job"""
    try:
        user_id = get_user_id()
        server_uri = get_server_uri()
        response = httpx.post(f"{server_uri}/fine-tune/{user_id}/{job_id}/stop")
        
        if response.status_code == 200:
            result = response.json()
            log_event("tuning_stopped", {"tune_id": job_id})
            typer.echo(f"Stopped fine-tuning job: {result['tune_id']}")
            typer.echo(f"Status: {result['status']}")
            typer.echo(f"Message: {result['message']}")
        else:
            log_event("tuning_stop_failed", {
                "reason": "api_error",
                "status_code": response.status_code,
                "tune_id": job_id
            })
            raise Exception(f"Failed to stop job: {response.text}")
            
    except Exception as e:
        log_event("tuning_stop_failed", {
            "reason": "exception",
            "error_type": type(e).__name__,
            "tune_id": job_id
        })
        typer.echo(f"Error: {str(e)}")
        raise typer.Exit(1)


@tune_app.command("logs")
@require_auth
def stream_logs(
    job_id: str = typer.Option(..., "-j", "--job-id", help="Job ID to stream logs for"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
):
    """Stream logs from a fine-tuning job"""
    log_event("tuning_logs_viewed", {"tune_id": job_id, "follow_mode": follow})
    user_id = get_user_id()
    server_uri = get_server_uri()
    
    try:
        last_size = 0
        first_read = True
        
        while True:
            # Only use range header after first read
            headers = {'Range': f'bytes={last_size}-'} if not first_read else {}
            
            with httpx.stream('GET', f"{server_uri}/fine-tune/{user_id}/{job_id}/log", headers=headers) as response:
                # Handle 416 (Range Not Satisfiable) for follow mode
                if response.status_code == 416:
                    if not follow:
                        break
                    sleep(5)
                    continue
                    
                response.raise_for_status()
                
                # Stream the content
                for chunk in response.iter_bytes():
                    if chunk:
                        sys.stdout.buffer.write(chunk)
                        sys.stdout.buffer.flush()
                        last_size += len(chunk)
            
            # Break if not following or after first read without follow
            if not follow and not first_read:
                break
                
            first_read = False
            sleep(5)
            
    except httpx.RequestError as e:
        typer.echo(f"Error connecting to API: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: {str(e)}", err=True)
        raise typer.Exit(1) 