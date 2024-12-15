#!/bin/bash

# Configuration
SERVER="http://localhost:8000"
EMAIL="test@example.com"
NAME="Test User"

# Function definitions
create_user() {
    echo "👤 Creating new user..."
    CREATE_RESPONSE=$(curl -s -X POST \
        "${SERVER}/auth/create_user" \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -d "{\"email\": \"${EMAIL}\", \"name\": \"${NAME}\"}" 2>&1)
    
    if [[ $CREATE_RESPONSE =~ "< HTTP/1.1 "[45] ]]; then
        echo "❌ Error creating user:"
        echo "$CREATE_RESPONSE"
        return 1
    fi

    USER_ID=$(echo "$CREATE_RESPONSE" | grep -A1 "^{" | jq -r '.user_id')
    if [ -z "$USER_ID" ]; then
        echo "❌ Failed to parse user ID from response:"
        echo "$CREATE_RESPONSE" 
        return 1
    fi

    echo "✅ User created. User ID: ${USER_ID}"
    echo "${USER_ID}"
}

create_token() {
    local user_id=$1
    echo "🔑 Creating token for user: ${user_id}" >&2
    TOKEN_RESPONSE=$(curl -s -X POST \
        "${SERVER}/auth/create_token?user_id=${user_id}" \
        -H 'accept: application/json')
    
    if [[ $TOKEN_RESPONSE =~ "< HTTP/1.1 "[45] ]]; then
        echo "❌ Error creating token:" >&2
        echo "$TOKEN_RESPONSE" >&2
        return 1
    fi

    TOKEN=$(echo "$TOKEN_RESPONSE" | grep -A1 "^{" | jq -r '.token')
    if [ -z "$TOKEN" ]; then
        echo "❌ Failed to parse token from response:" >&2
        echo "$TOKEN_RESPONSE" >&2
        return 1
    fi

    echo "✅ Token created: ${TOKEN}" >&2
    echo "${TOKEN}"
}

login() {
    local token=$1
    echo "🔓 Logging in with token..." >&2
    curl -s -X POST \
        "${SERVER}/auth/login" \
        -H 'Content-Type: application/json' \
        -d "{\"token\": \"${token}\"}" | jq '.'
}

run_full_test() {
    echo "🚀 Running full auth API test"
    echo "------------------------"

    # Create user and get user ID
    USER_ID=$(create_user | tail -n 1)
    echo "User ID: ${USER_ID}"
    echo

    # Create token for user
    TOKEN=$(create_token "${USER_ID}")
    echo "Token: ${TOKEN}"
    echo

    # Login with token
    echo "Login response:"
    login $TOKEN
    echo

    echo "✨ All operations completed!"
}

# Command handling
case "$1" in
    "create_user")
        create_user
        ;;
    "create_token")
        if [ -z "$2" ]; then
            echo "Error: User ID required for token creation"
            exit 1
        fi
        create_token $2
        ;;
    "login")
        if [ -z "$2" ]; then
            echo "Error: Token required for login"
            exit 1
        fi
        login $2
        ;;
    *)
        run_full_test
        ;;
esac 