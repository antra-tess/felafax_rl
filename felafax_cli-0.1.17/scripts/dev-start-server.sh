#!/bin/bash

# Check if --no-reload flag is provided
if [[ "$*" == *"--no-reload"* ]]; then
    # Run without hot reloading
    uvicorn felafax.server.main:app --host 0.0.0.0 --port 8000 --log-level info
else
    # Run with hot reloading enabled
    uvicorn felafax.server.main:app --reload --host 0.0.0.0 --port 8000 --log-level info
fi
