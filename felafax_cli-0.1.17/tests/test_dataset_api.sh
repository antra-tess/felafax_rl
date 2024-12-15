#!/bin/bash

# Configuration
SERVER="http://localhost:8000"
USER_ID="b4c9a289323b"
DATASET_FILE="dataset_small.json"

# Function definitions
upload_single() {
    echo "📤 Testing single file upload..."
    UPLOAD_RESPONSE=$(curl -s -X POST \
        "${SERVER}/datasets/${USER_ID}/upload" \
        -H 'accept: application/json' \
        -H 'Content-Type: multipart/form-data' \
        -F "file=@${DATASET_FILE}")

    DATASET_ID=$(echo $UPLOAD_RESPONSE | grep -o '"dataset_id":"[^"]*' | cut -d'"' -f4)
    echo "✅ Upload complete. Dataset ID: ${DATASET_ID}"
    echo $DATASET_ID  # Return dataset ID
}

upload_chunked() {
    echo "📦 Testing chunked upload..."
    UPLOAD_ID="upload_$(date +%s)"
    split -n 3 "${DATASET_FILE}" "${DATASET_FILE}_chunk_"

    # Upload first chunk
    echo "Uploading chunk 0..."
    curl -s -X POST \
        "${SERVER}/datasets/${USER_ID}/upload/chunked" \
        -H 'accept: application/json' \
        -H 'Content-Type: multipart/form-data' \
        -F "file=@${DATASET_FILE}_chunk_aa" \
        -F "upload_id=${UPLOAD_ID}" \
        -F "chunk_number=0" \
        -F "total_chunks=3"

    # Upload second chunk
    echo "Uploading chunk 1..."
    curl -s -X POST \
        "${SERVER}/datasets/${USER_ID}/upload/chunked" \
        -H 'accept: application/json' \
        -H 'Content-Type: multipart/form-data' \
        -F "file=@${DATASET_FILE}_chunk_ab" \
        -F "upload_id=${UPLOAD_ID}" \
        -F "chunk_number=1" \
        -F "total_chunks=3"

    # Upload final chunk
    echo "Uploading chunk 2..."
    CHUNKED_RESPONSE=$(curl -s -X POST \
        "${SERVER}/datasets/${USER_ID}/upload/chunked" \
        -H 'accept: application/json' \
        -H 'Content-Type: multipart/form-data' \
        -F "file=@${DATASET_FILE}_chunk_ac" \
        -F "upload_id=${UPLOAD_ID}" \
        -F "chunk_number=2" \
        -F "total_chunks=3")

    CHUNKED_DATASET_ID=$(echo $CHUNKED_RESPONSE | grep -o '"dataset_id":"[^"]*' | cut -d'"' -f4)
    echo "✅ Chunked upload complete. Dataset ID: ${CHUNKED_DATASET_ID}"
    # Cleanup chunk files
    rm "${DATASET_FILE}_chunk_"*
    echo $CHUNKED_DATASET_ID  # Return dataset ID
}

list_datasets() {
    echo "📋 Listing all datasets..."
    curl -s -X GET \
        "${SERVER}/datasets/${USER_ID}/list" \
        -H 'accept: application/json' | jq '.'
}

delete_dataset() {
    local dataset_id=$1
    echo "🗑️  Deleting dataset: $dataset_id"
    curl -s -X DELETE \
        "${SERVER}/datasets/${USER_ID}/${dataset_id}" \
        -H 'accept: application/json'
}

run_full_test() {
    echo "🚀 Running full dataset API test"
    echo "------------------------"

    # Run single upload
    SINGLE_ID=$(upload_single)
    echo

    # Run chunked upload
    CHUNKED_ID=$(upload_chunked)
    echo

    # List datasets
    list_datasets
    echo

    # Delete datasets
    for id in "$SINGLE_ID" "$CHUNKED_ID"; do
        if [ ! -z "$id" ]; then
            delete_dataset $id
            echo
        fi
    done

    echo "✨ All operations completed!"
}

# Command handling
case "$1" in
    "single")
        upload_single
        ;;
    "chunked")
        upload_chunked
        ;;
    "list")
        list_datasets
        ;;
    "delete")
        if [ -z "$2" ]; then
            echo "Error: Dataset ID required for delete command"
            exit 1
        fi
        delete_dataset $2
        ;;
    *)
        run_full_test
        ;;
esac
