#!/bin/bash

# Configuration
SERVER="http://localhost:8000"
USER_ID="b4c9a289323b"
MODEL_NAME="llama3-70b"
DOCKER_IMAGE="gcr.io/felafax-training/vllm:latest_v3"
DISK_SIZE_GB=1000
ATTACH_DISK=true
# DOCKER_ENV='{"HF_PATH": "NousResearch/Meta-Llama-3-8B-Instruct"}'
DOCKER_ENV='{"GCS_MODEL_PATH": "gs://felafax-storage/MODEL_STORAGE/models--NousResearch--Meta-Llama-3-8B-Instruct/"}'

# Function definitions
start_accelerator() {
    echo "🚀 Starting accelerator..."
    START_RESPONSE=$(curl -s -X POST \
        "${SERVER}/accelerators/${USER_ID}/start" \
        -H 'Content-Type: application/json' \
        -d "{
            \"model_name\": \"${MODEL_NAME}\",
            \"docker_image\": \"${DOCKER_IMAGE}\",
            \"attach_disk\": ${ATTACH_DISK},
            \"disk_size_gb\": ${DISK_SIZE_GB},
            \"docker_env\": ${DOCKER_ENV}
        }")
    echo $START_RESPONSE
    
    ACCELERATOR_ID=$(echo $START_RESPONSE | grep -o '"accelerator_id":"[^"]*' | cut -d'"' -f4)
    echo "✅ Accelerator started. Accelerator ID: ${ACCELERATOR_ID}"
    echo $ACCELERATOR_ID  # Return accelerator ID
}

check_status() {
    local accelerator_id=$1
    echo "📊 Checking accelerator status..."
    curl -s -X GET \
        "${SERVER}/accelerators/${USER_ID}/${accelerator_id}/status" \
        -H 'accept: application/json' | jq '.'
}

stop_accelerator() {
    local accelerator_id=$1
    echo "🛑 Stopping accelerator..."
    curl -s -X POST \
        "${SERVER}/accelerators/${USER_ID}/${accelerator_id}/stop" \
        -H 'accept: application/json' | jq '.'
}

list_accelerators() {
    echo "📋 Listing all accelerators..."
    curl -s -X GET \
        "${SERVER}/accelerators/${USER_ID}/list" \
        -H 'accept: application/json' | jq '.'
}

delete_accelerator() {
    local accelerator_id=$1
    echo "🗑️  Deleting accelerator..."
    curl -s -X DELETE \
        "${SERVER}/accelerators/${USER_ID}/${accelerator_id}" \
        -H 'accept: application/json' | jq '.'
}

run_full_test() {
    echo "🚀 Running full accelerator API test"
    echo "------------------------"

    # Start and get accelerator ID
    ACCELERATOR_ID=$(start_accelerator)
    echo

    # Initial status check
    check_status $ACCELERATOR_ID
    echo

    # List all accelerators
    list_accelerators
    echo

    # Wait and check again
    echo "⏳ Waiting for 10 seconds..."
    sleep 10
    check_status $ACCELERATOR_ID
    echo

    # Stop accelerator
    stop_accelerator $ACCELERATOR_ID
    echo

    # Final status check
    check_status $ACCELERATOR_ID
    echo

    # Delete accelerator
    delete_accelerator $ACCELERATOR_ID
    echo

    echo "✨ All operations completed!"
}

# Command handling
case "$1" in
    "start")
        start_accelerator
        ;;
    "status")
        if [ -z "$2" ]; then
            echo "Error: Accelerator ID required for status check"
            exit 1
        fi
        check_status $2
        ;;
    "stop")
        if [ -z "$2" ]; then
            echo "Error: Accelerator ID required for stop command"
            exit 1
        fi
        stop_accelerator $2
        ;;
    "list")
        list_accelerators
        ;;
    "delete")
        if [ -z "$2" ]; then
            echo "Error: Accelerator ID required for delete command"
            exit 1
        fi
        delete_accelerator $2
        ;;
    *)
        run_full_test
        ;;
esac 