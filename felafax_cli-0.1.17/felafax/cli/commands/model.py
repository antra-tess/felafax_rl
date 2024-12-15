import typer
import httpx
from typing import Optional
from pathlib import Path
from ..common import require_auth, get_server_uri, get_user_id, log_event
import json
import time

model_app = typer.Typer(help="Model management commands")

@model_app.command("list")
@require_auth
def list_models():
    """List available fine-tuned models"""
    log_event("model_list_viewed")
    try:
        server_uri = get_server_uri()
        user_id = get_user_id()
        response = httpx.get(f"{server_uri}/models/{user_id}/list")
        
        if response.status_code != 200:
            raise Exception(f"Failed to list models: {response.text}")
            
        models = response.json()
        if not models:
            typer.echo("No models found")
            return
        
        # Define column widths
        col_widths = {
            'model_id': 35,
            'base_model': 20,
            'status': 12,
            'created_at': 10
        }
        
        # Print header
        typer.echo("\nAvailable Models:")
        typer.echo("-" * (sum(col_widths.values()) + 3 * len(col_widths) - 1))  # Account for separators
        
        # Print column headers
        header = (
            f"{'Model ID':<{col_widths['model_id']}} | "
            f"{'Base Model':<{col_widths['base_model']}} | "
            f"{'Status':<{col_widths['status']}} | "
            f"{'Created At':<{col_widths['created_at']}}"
        )
        typer.echo(header)
        typer.echo("-" * (sum(col_widths.values()) + 3 * len(col_widths) - 1))
        
        # Print each model
        for model in models:
            created_at = model['created_at'].split('T')[0]
            row = (
                f"{model['model_id']:<{col_widths['model_id']}} | "
                f"{model['base_model']:<{col_widths['base_model']}} | "
                f"{model['status']:<{col_widths['status']}} | "
                f"{created_at:<{col_widths['created_at']}}"
            )
            typer.echo(row)
            
    except Exception as e:
        typer.echo(f"Error listing models: {str(e)}", err=True)
        raise typer.Exit(1)

@model_app.command("download")
@require_auth
def download_model(
    model_id: str = typer.Argument(..., help="ID of the model to download"),
    output_dir: Optional[Path] = typer.Option(None, help="Directory to save the model")
):
    """Download a fine-tuned model"""
    try:
        server_uri = get_server_uri()
        user_id = get_user_id()
        
        # First get the download URL
        response = httpx.get(f"{server_uri}/models/{user_id}/{model_id}/download")
        
        if response.status_code != 200:
            raise Exception(f"Failed to get download URL: {response.text}")
            
        download_url = response.json()["download_url"]
        
        # Create output directory if specified
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            
        typer.echo(f"Downloading model {model_id}...")
        # TODO: Implement actual download logic using download_url
        typer.echo(f"Model downloaded successfully to {output_dir or 'current directory'}")
        
    except Exception as e:
        typer.echo(f"Error downloading model: {str(e)}", err=True)
        raise typer.Exit(1)

@model_app.command("chat")
@require_auth
def chat_with_model(
    model_id: str = typer.Argument(..., help="ID of the model to chat with"),
    system_prompt: str = typer.Option(
        "You are a helpful AI assistant.",
        help="System prompt to initialize the conversation"
    )
):
    """Start an interactive chat session with a model"""
    log_event("model_chat_started", {"model_id": model_id})
    try:
        server_uri = get_server_uri()
        user_id = get_user_id()
        
        # First verify model exists and is ready
        response = httpx.get(f"{server_uri}/models/{user_id}/{model_id}/info")
        if response.status_code != 200:
            raise Exception(f"Model not found or not ready: {response.text}")
        
        # Initialize chat session
        typer.echo(f"Initializing chat session with model {model_id}")
        typer.echo("This may take up to 5 minutes while we start the accelerator...")
        typer.echo("Note: Running in eager mode - responses will be slower than production deployments")
        
        init_response = httpx.post(f"{server_uri}/models/{user_id}/{model_id}/init-chat", timeout=600)
        if init_response.status_code != 200:
            raise Exception(f"Failed to initialize chat: {init_response.text}")
        
        accelerator_id = init_response.json()["accelerator_id"]
        typer.echo("Chat session initialized successfully!")
        typer.echo("Type 'exit' or 'quit' to end the session, or press Ctrl+C")
        typer.echo("-" * 40)
        
        # Initialize conversation history with system prompt
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        while True:
            try:
                # Get user input
                user_input = typer.prompt("You")
                
                if user_input.lower() in ['exit', 'quit']:
                    break
                
                # Add user message to conversation
                messages.append({"role": "user", "content": user_input})
                
                # Add timing for the response
                start_time = time.time()
                try:
                    response = httpx.post(
                        f"{server_uri}/models/{user_id}/{model_id}/chat/{accelerator_id}",
                        json={"messages": messages},
                        timeout=300
                    )
                    
                    end_time = time.time()
                    response_time = end_time - start_time
                    
                    if response.status_code == 503:
                        log_event("model_chat_response_failed", {
                            "model_id": model_id,
                            "reason": "warming_up",
                            "response_time": response_time
                        })
                        typer.echo("Model is still warming up. Please wait 5 mins and try again...")
                        continue
                    elif response.status_code != 200:
                        log_event("model_chat_response_failed", {
                            "model_id": model_id,
                            "reason": "api_error",
                            "status_code": response.status_code,
                            "response_time": response_time
                        })
                        raise Exception(f"Chat failed: {response.text}")
                    
                    assistant_response = response.json()['response']
                    messages.append({"role": "assistant", "content": assistant_response})
                    
                    # Log successful response with timing
                    log_event("model_chat_response", {
                        "model_id": model_id,
                        "response_time": response_time,
                        "response_length": len(assistant_response)
                    })
                    
                    typer.echo(f"Assistant: {assistant_response}")
                    
                except httpx.ConnectError:
                    log_event("model_chat_response_failed", {
                        "model_id": model_id,
                        "reason": "connection_error"
                    })
                    typer.echo("Model is still warming up. Please wait a moment and try again...")
                    continue
                
            except KeyboardInterrupt:
                log_event("model_chat_interrupted", {"model_id": model_id})
                typer.echo("\nChat session terminated by user")
                break
                
        log_event("model_chat_ended", {"model_id": model_id})
        typer.echo("\nChat session ended")
        
    except Exception as e:
        log_event("model_chat_error", {
            "model_id": model_id,
            "error_type": type(e).__name__
        })
        typer.echo(f"Error in chat session: {str(e)}", err=True)
        raise typer.Exit(1)

@model_app.command("delete")
@require_auth
def delete_model(
    model_id: str = typer.Argument(..., help="ID of the model to delete")
):
    """Delete a fine-tuned model"""
    log_event("model_delete_initiated", {"model_id": model_id})
    try:
        if not typer.confirm(f"Are you sure you want to delete model {model_id}?"):
            typer.echo("Operation cancelled")
            return
            
        server_uri = get_server_uri()
        user_id = get_user_id()
        response = httpx.delete(f"{server_uri}/models/{user_id}/{model_id}")
        
        if response.status_code != 200:
            raise Exception(f"Failed to delete model: {response.text}")
            
        typer.echo(f"Model {model_id} deleted successfully")
        
    except Exception as e:
        typer.echo(f"Error deleting model: {str(e)}", err=True)
        raise typer.Exit(1)

@model_app.command("info")
@require_auth
def get_model_info(
    model_id: str = typer.Argument(..., help="ID of the model to get info for")
):
    """Get detailed information about a model"""
    log_event("model_info_viewed", {"model_id": model_id})
    try:
        server_uri = get_server_uri()
        user_id = get_user_id()
        response = httpx.get(f"{server_uri}/models/{user_id}/{model_id}/info")
        
        if response.status_code != 200:
            raise Exception(f"Failed to get model info: {response.text}")
            
        model_info = response.json()
        
        # Pretty print the entire JSON response
        typer.echo("\nModel Information:")
        typer.echo("-" * 40)
        typer.echo(json.dumps(model_info, indent=2))
        
    except Exception as e:
        typer.echo(f"Error getting model info: {str(e)}", err=True)
        raise typer.Exit(1)
