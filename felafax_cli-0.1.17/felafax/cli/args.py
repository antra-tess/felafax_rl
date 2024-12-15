import typer

# Training command arguments
training_args = {
    "config_path": typer.Argument(
        ...,
        help="Path to training config YAML",
        file_okay=True,
        dir_okay=False,
        exists=True,
        readable=True,
    ),
    "output_dir": typer.Option(
        None,
       help="Override output directory",
        dir_okay=True,
        file_okay=False,
        resolve_path=True,
    ),
    "tpu_name": typer.Option(
        None,
        help="Override TPU name",
    )
}

# Accelerator command arguments
accelerator_start_args = {
    "name": typer.Argument(
        ...,
        help="Accelerator node name"
    ),
    "location": typer.Option(
        "us-central1-a",
        help="Cloud location (zone/region)"
    ),
    "accelerator_type": typer.Option(
        "tpu",
        help="Accelerator type (tpu, gpu, etc)"
    ),
    "tpu_type": typer.Option(
        "v3",
        help="TPU family type: v3, v4, v5p, v5e"
    ),
    "tpu_size": typer.Option(
        8,
        help="Number of TPU cores: 8, 16, 32, 64, 128, 256, 512, 1024"
    )
}

accelerator_stop_args = {
    "name": typer.Argument(
        ...,
        help="Accelerator node name"
    ),
    "location": typer.Option(
        "us-central1-a",
        help="Cloud location (zone/region)"
    ),
    "accelerator_type": typer.Option(
        "tpu",
        help="Accelerator type (tpu, gpu, etc)"
    )
}
