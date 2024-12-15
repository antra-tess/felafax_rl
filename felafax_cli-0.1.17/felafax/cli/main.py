import typer
from importlib.metadata import version
from .log import setup_logger
from .commands.tune import tune_app
from .commands.files import files_app
from .commands.auth import auth_app
from .commands.model import model_app
from .commands.accelerator import accelerator_app
import logging
from .common import init_posthog

def version_callback(show_version: bool):
    if show_version:
        try:
            print(f"Felafax CLI Version: {version('felafax-cli')}")
        except Exception:
            print("Version information not available")
        raise typer.Exit()

# Create typer app
logger = logging.getLogger(__name__)
setup_logger()
init_posthog()

# Create sub-commands
app = typer.Typer(help="Felafax CLI")

@app.callback()
def main(version: bool = typer.Option(None, "--version", "-v", help="Show version and exit.", callback=version_callback, is_eager=True)):
    """
    Felafax CLI tool
    """
    pass

app.add_typer(auth_app, name="auth")
app.add_typer(files_app, name="files")
app.add_typer(tune_app, name="tune")
app.add_typer(model_app, name="model")
app.add_typer(accelerator_app, name="accelerator", hidden=True)

if __name__ == "__main__":
    logger.info("Starting Felafax CLI")
    app()