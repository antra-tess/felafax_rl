#!/bin/bash
set -eo pipefail

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="gcr.io/felafax-training/trainer"
IMAGE_TAG="latest"
GCS_JOB_PATH="gs://felafax-storage-v2/users/b4c9a289323b/finetunes/tune_8eb7b3aa1bdb"
SCRIPT_NAME="finetune_runner.py"

echo -e "${GREEN}Building trainer Docker image locally...${NC}"

# Navigate to the project root and copy key file
PROJECT_ROOT="$(pwd)"
cp "$PROJECT_ROOT/key.json" felafax/remote/

# Change to the directory containing Dockerfile.trainer
cd "felafax/remote"

# Build the Docker image for local platform
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" -f Dockerfile.trainer .

# Clean up the copied key file
rm key.json

echo -e "${GREEN}Running trainer container...${NC}"
docker run \
    --privileged \
    --cap-add SYS_ADMIN \
    --device /dev/fuse \
    --security-opt apparmor:unconfined \
    -e GCS_JOB_PATH="${GCS_JOB_PATH}" \
    -e SCRIPT_NAME="${SCRIPT_NAME}" \
    "${IMAGE_NAME}:${IMAGE_TAG}"

cd "$PROJECT_ROOT" 