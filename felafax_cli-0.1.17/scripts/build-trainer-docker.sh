#!/bin/bash
set -eo pipefail

# Script configuration
IMAGE_NAME="gcr.io/felafax-training/trainer"
IMAGE_TAG="latest"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building trainer Docker image...${NC}"

PROJECT_ROOT="$(pwd)"
cp "$PROJECT_ROOT/key.json" felafax/remote/

cd "felafax/remote"

docker build --platform linux/amd64 -t "${IMAGE_NAME}:${IMAGE_TAG}" -f Dockerfile.trainer .

# Clean up the copied key file
rm key.json

# Push the image to Google Container Registry
echo -e "${GREEN}Pushing image to GCR...${NC}"
docker push "${IMAGE_NAME}:${IMAGE_TAG}"

# Sync scripts to GCS
echo -e "${GREEN}Syncing scripts to GCS...${NC}"
cd "$PROJECT_ROOT"
