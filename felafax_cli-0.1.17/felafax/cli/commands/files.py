import typer
import httpx
from pathlib import Path
from typing import Optional
from ..common import require_auth, get_server_uri, get_user_id, log_event

files_app = typer.Typer(help="File management commands")

@files_app.command("list")
@require_auth
def list_files(
    prefix: Optional[str] = typer.Option(None, help="Filter files by prefix"),
    limit: int = typer.Option(100, help="Maximum number of files to list")
):
    """List files in your storage"""
    try:
        server_uri = get_server_uri()
        user_id = get_user_id()
        response = httpx.get(f"{server_uri}/datasets/{user_id}/list")
        
        if response.status_code != 200:
            raise Exception(f"Failed to list files: {response.text}")
            
        files = response.json()
        if not files:
            typer.echo("No files found")
            return
        
        # Define column widths
        col_widths = {
            'dataset_id': 35,
            'name': 30,
            'size': 15,
            'created_at': 10
        }
        
        # Print header
        typer.echo("\nAvailable Datasets:")
        typer.echo("-" * (sum(col_widths.values()) + 3 * len(col_widths) - 1))
        
        # Print column headers
        header = (
            f"{'Dataset ID':<{col_widths['dataset_id']}} | "
            f"{'Name':<{col_widths['name']}} | "
            f"{'Size':<{col_widths['size']}} | "
            f"{'Created At':<{col_widths['created_at']}}"
        )
        typer.echo(header)
        typer.echo("-" * (sum(col_widths.values()) + 3 * len(col_widths) - 1))
        
        # Print each file
        for file in files:
            # Format size to be human readable
            size_bytes = file['size_bytes']
            if size_bytes < 1024:
                size_str = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                size_str = f"{size_bytes/1024:.1f} KB"
            elif size_bytes < 1024 * 1024 * 1024:
                size_str = f"{size_bytes/(1024*1024):.1f} MB"
            else:
                size_str = f"{size_bytes/(1024*1024*1024):.1f} GB"
            
            # Format created_at if available, otherwise use empty string
            created_at = file.get('created_at', '').split('T')[0] if 'created_at' in file else ''
            
            row = (
                f"{file['dataset_id']:<{col_widths['dataset_id']}} | "
                f"{file['name']:<{col_widths['name']}} | "
                f"{size_str:<{col_widths['size']}} | "
                f"{created_at:<{col_widths['created_at']}}"
            )
            typer.echo(row)
            
    except Exception as e:
        typer.echo(f"Error listing files: {str(e)}", err=True)
        raise typer.Exit(1)

@files_app.command("upload")
@require_auth
def upload_file(
    file_path: Path = typer.Argument(..., help="Path to the file to upload"),
):
    """Upload a file to storage"""
    if not file_path.exists():
        log_event("file_upload_failed", {"reason": "file_not_found", "path": str(file_path)})
        typer.echo(f"File not found: {file_path}")
        raise typer.Exit(1)
    
    try:
        server_uri = get_server_uri()
        user_id = get_user_id()
        
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f)}
            response = httpx.post(
                f"{server_uri}/datasets/{user_id}/upload",
                files=files
            )
        
        if response.status_code != 200:
            log_event("file_upload_failed", {"reason": "server_error", "status_code": response.status_code})
            raise Exception(f"Upload failed: {response.text}")
            
        result = response.json()
        log_event("file_upload_successful", {
            "file_name": file_path.name,
            "dataset_id": result['dataset_id']
        })
        typer.echo(f"File uploaded.\nDataset ID: {result['dataset_id']}")
    except Exception as e:
        log_event("file_upload_failed", {
            "reason": "exception",
            "error_type": type(e).__name__
        })
        typer.echo(f"Error uploading file: {str(e)}", err=True)
        raise typer.Exit(1)

@files_app.command("delete")
@require_auth
def delete_file(
    file_path: str = typer.Argument(..., help="Path to the file to delete")
):
    """Delete a file from storage"""
    try:
        server_uri = get_server_uri()
        user_id = get_user_id()
        response = httpx.delete(f"{server_uri}/datasets/{user_id}/{file_path}")
        
        if response.status_code != 200:
            log_event("file_delete_failed", {"reason": "server_error", "status_code": response.status_code})
            raise Exception(f"Delete failed: {response.text}")
        
        log_event("file_delete_successful", {"file_path": file_path})    
        typer.echo(f"File deleted successfully: {file_path}")
    except Exception as e:
        log_event("file_delete_failed", {
            "reason": "exception",
            "error_type": type(e).__name__
        })
        typer.echo(f"Error deleting file: {str(e)}", err=True)
        raise typer.Exit(1)