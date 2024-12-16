#!/bin/bash
# This script installs uv on the current host

# Make sure pip is available. On TPU VMs, pip usually comes with the system Python.
# If you are using python3 explicitly:
# python3 -m pip install --upgrade pip
# python3 -m pip install uv
# Or just:
pip install --upgrade pip
pip install uv

