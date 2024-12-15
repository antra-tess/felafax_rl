#!/bin/bash
set -eo pipefail

SCRIPTS_BUCKET="gs://felafax-storage-v2/scripts"

# First, remove any existing scripts in the bucket
# gsutil -m rm -r "${SCRIPTS_BUCKET}/*" || true

# Copy all scripts and requirements.txt to GCS
gsutil -m cp -r felafax/remote/scripts/* "${SCRIPTS_BUCKET}/"

echo -e "${GREEN}Build and sync completed successfully!${NC}"