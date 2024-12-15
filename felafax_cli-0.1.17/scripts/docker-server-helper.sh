#!/bin/bash

# Set error handling
set -e

# Common variables
IMAGE_NAME="gcr.io/felafax-training/api-server"
SERVER_NAME="felafax-api-server"
REGION="us-central1-f"
LOCAL_CONTAINER_NAME="felafax-api-server-local"

# Function to build Docker image
build() {
    local platform=$1
    echo "Building docker image..."
    if [ "$platform" == "arm64" ]; then
        echo "Building for Apple Silicon (ARM64)..."
        docker build -t $IMAGE_NAME:latest --platform linux/arm64 .
    else
        echo "Building for default platform..."
        docker build -t $IMAGE_NAME:latest .
    fi
}

# Function to run locally
run_local() {
    echo "Building for Apple Silicon first..."
    build "arm64"
    
    echo "Stopping any existing local containers..."
    docker rm -f $LOCAL_CONTAINER_NAME 2>/dev/null || true
    
    echo "Running container locally..."
    docker run -d \
        --name $LOCAL_CONTAINER_NAME \
        -p 8000:8000 \
        $IMAGE_NAME:latest

    echo "Checking container status..."
    docker ps

    echo "Waiting for container to start..."
    sleep 5

    echo "Testing API..."
    curl -X POST http://localhost:8000/auth/create_user \
        -H "Content-Type: application/json" \
        -d '{"email": "test@example.com", "name": "Test User"}'

    echo -e "\nContainer logs:"
    docker logs $LOCAL_CONTAINER_NAME
}

# Function to run locally with logs in foreground
run_local_fg() {
    echo "Building for Apple Silicon first..."
    build "arm64"
    
    echo "Stopping any existing local containers..."
    docker rm -f $LOCAL_CONTAINER_NAME 2>/dev/null || true
    
    echo "Running container locally with logs..."
    docker run \
        --name $LOCAL_CONTAINER_NAME \
        -p 8000:8000 \
        $IMAGE_NAME:latest
}

# Function to build and push to server
build_and_push() {
    echo "Building docker image..."
    build

    echo "Pushing docker image... $IMAGE_NAME"
    docker push $IMAGE_NAME:latest

    echo "Prune docker..."
    gcloud compute ssh $SERVER_NAME --zone=$REGION --command="
        docker system prune -af --volumes
    "

    echo "Pulling the new container..."
    gcloud compute ssh $SERVER_NAME --zone=$REGION --command="
        docker pull $IMAGE_NAME:latest
    "

    echo "Stopping and removing existing containers..."
    gcloud compute ssh $SERVER_NAME --zone=$REGION --command="
        docker ps -q | xargs -r docker stop || true &&
        docker ps -aq | xargs -r docker rm -f || true
    "

    echo "Starting new container..."
    gcloud compute ssh $SERVER_NAME --zone=$REGION --command="
        docker run -d --name $SERVER_NAME -p 8000:8000 $IMAGE_NAME:latest
    "

    echo "Waiting for container to start..."
    sleep 10

    echo "Verifying running Docker containers..."
    gcloud compute ssh $SERVER_NAME --zone=$REGION --command="docker ps"

    echo "Getting IP address..."
    ip_address=$(gcloud compute instances describe $SERVER_NAME --zone=$REGION --format="get(networkInterfaces[0].accessConfigs[0].natIP)")
    echo "Server is running at $ip_address"
}

# Main script logic
case "$1" in
    "build")
        if [ "$2" == "arm64" ]; then
            build "arm64"
        else
            build
        fi
        ;;
    "test")
        run_local
        ;;
    "run")
        run_local_fg
        ;;
    "deploy")
        build_and_push
        ;;
    *)
        echo "Usage: $0 {build|test|run|deploy}"
        echo "  build [arm64] - Build the Docker image (optionally for ARM64)"
        echo "  test         - Run the container locally and test"
        echo "  run          - Run the container locally with logs in foreground"
        echo "  deploy       - Build, push and deploy to server"
        exit 1
        ;;
esac