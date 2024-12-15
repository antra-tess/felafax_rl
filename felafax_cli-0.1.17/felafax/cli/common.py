import os
import json
import typer
from typing import Dict, Any, Optional
from functools import wraps
import asyncio
from felafax.config import Config
from posthog import Posthog

def load_config() -> Dict:
    """Load user configuration from file"""
    if os.path.exists(Config.CONFIG_FILE):
        with open(Config.CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_config(config: Dict) -> None:
    """Save user configuration to file"""
    os.makedirs(Config.CONFIG_DIR, exist_ok=True)
    with open(Config.CONFIG_FILE, 'w') as f:
        json.dump(config, f)

def get_user_id() -> str:
    """Get the user ID from the configuration"""
    config = load_config()
    return config.get("user_id")

def require_auth(f):
    """Decorator to ensure user is authenticated before running commands"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        config = load_config()
        if not config.get("token"):
            typer.echo("Please login first using 'felafax login'")
            raise typer.Exit(1)
        return f(*args, **kwargs)
    return wrapper

def async_command(f):
    """Decorator to handle async commands properly"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(f(*args, **kwargs))
    return wrapper 

def get_server_uri() -> str:
    """Get the server URI based on environment settings"""
    if Config.DEBUG or Config.DEV:
        return Config.DEV_URL
    return Config.PROD_URL

def init_posthog():
    # Create PostHog client as a global variable
    global posthog_client
    posthog_client = Posthog(
        project_api_key='phc_NO1HwOFphIGgua2l6GdPb2a9nqvBBcEDOYCfrzVsWiY',
    host='https://us.i.posthog.com'
)

def log_event(event_name: str, properties: Optional[Dict[str, Any]] = None):
    """Log an event to PostHog with the current user ID"""
    try:
        config = load_config()
        user_id = config.get("user_id", "anonymous")
        posthog_client.capture(distinct_id=user_id, event=event_name, properties=properties or {})
    except Exception:
        # Silently fail if logging fails - we don't want to interrupt the CLI
        pass