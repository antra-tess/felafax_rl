import typer
import httpx
from ..common import load_config, save_config, get_server_uri, log_event

auth_app = typer.Typer(help="Authentication commands")

@auth_app.command("login")
def login(
    token: str = typer.Option(..., help="Authentication token"),
    force: bool = typer.Option(False, "--force", "-f", help="Force login even if already logged in")
):
    """Login to Felafax using an authentication token"""
    config = load_config()
    
    # Check if already logged in
    if not force and config.get("token"):
        typer.echo("Already logged in. Use --force to login again.")
        raise typer.Exit(1)
    
    try:
        # Call the server API to validate token and get user_id
        server_uri = get_server_uri()
        response = httpx.post(f"{server_uri}/auth/login", json={"token": token})
        
        if response.status_code != 200:
            log_event("login_failed", {
                "reason": "invalid_token",
                "status_code": response.status_code
            })
            raise Exception("Invalid token")
            
        response_data = response.json()
        user_id = response_data["user_id"]
        token = response_data["token"]
        
        if not user_id or not token:
            log_event("login_failed", {"reason": "missing_credentials"})
            raise Exception("Login failed, no user_id or token returned")
        
        config.update({
            "token": token,
            "user_id": user_id
        })
        save_config(config)
        log_event("login_successful", {"user_id": user_id})
        typer.echo(f"Successfully logged in as user {user_id}")
    except Exception as e:
        log_event("login_failed", {
            "reason": "exception",
            "error": str(e),
            "error_type": type(e).__name__
        })
        typer.echo(f"Login failed: {str(e)}", err=True)
        raise typer.Exit(1)

@auth_app.command("logout")
def logout():
    """Logout from Felafax"""
    config = load_config()
    
    if not config.get("token"):
        log_event("logout_skipped", {"reason": "not_logged_in"})
        typer.echo("Not logged in.")
        return
    
    user_id = config.get("user_id")
    config.pop("token", None)
    config.pop("user_id", None)
    save_config(config)
    log_event("logout_successful", {"user_id": user_id})
    typer.echo("Successfully logged out")

@auth_app.command("reset", hidden=True)
def reset(
    force: bool = typer.Option(False, "--force", "-f", help="Force reset without confirmation")
):
    """Reset all user data and metadata"""
    config = load_config()
    
    # Check if logged in
    if not config.get("token"):
        typer.echo("Not logged in.")
        raise typer.Exit(1)
    
    # Get confirmation unless force flag is used
    if not force:
        confirm = typer.confirm("This will delete all your data. Are you sure?")
        if not confirm:
            typer.echo("Reset cancelled.")
            raise typer.Exit(0)
    
    try:
        # Call the server API to reset user data
        server_uri = get_server_uri()
        response = httpx.post(
            f"{server_uri}/auth/reset",
            params={"user_id": config["user_id"]}
        )
        
        if response.status_code != 200:
            log_event("reset_failed", {
                "reason": "api_error",
                "status_code": response.status_code
            })
            raise Exception(f"Reset failed: {response.text}")
            
        # Clear local config
        config.pop("token", None)
        config.pop("user_id", None)
        save_config(config)
        
        log_event("reset_successful", {"user_id": config.get("user_id")})
        typer.echo("Successfully reset all user data and logged out")
        
    except Exception as e:
        log_event("reset_failed", {
            "reason": "exception",
            "error": str(e),
            "error_type": type(e).__name__
        })
        typer.echo(f"Reset failed: {str(e)}", err=True)
        raise typer.Exit(1) 