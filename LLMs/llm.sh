#!/bin/bash

# --- Configuration ---
SERVER_URL="http://localhost:5000" # Your server address and port
PREDICT_ENDPOINT="${SERVER_URL}/predict"
HEALTH_ENDPOINT="${SERVER_URL}/health"

# --- Helper Functions ---
function show_usage() {
    echo "Usage: $0 \"Your question here\""
    echo "Example: $0 \"What is the capital of France?\""
    exit 1
}

# Check if jq is installed for prettier JSON output (optional)
JQ_EXISTS=$(command -v jq)

# --- Main Script ---

# Check if a question was provided
if [ -z "$1" ]; then
    echo "Error: No question provided."
    show_usage
fi

# The question is the first argument (and potentially subsequent arguments if not quoted)
# It's best to pass the question in quotes if it contains spaces.
QUESTION_TEXT="$*" # Takes all arguments as a single string

# Optional: Check server health first
# echo "Checking server health..."
# health_status_code=$(curl -s -o /dev/null -w "%{http_code}" "${HEALTH_ENDPOINT}")
# if [ "$health_status_code" -ne 200 ]; then
#     echo "Error: Server at ${HEALTH_ENDPOINT} is not healthy (HTTP status: ${health_status_code}). Aborting."
#     curl -s "${HEALTH_ENDPOINT}" # Show error from server if any
#     exit 1
# else
#     echo "Server is healthy."
# fi

echo "Sending question to model: \"${QUESTION_TEXT}\""

# Construct the JSON payload
# Using jq is safer for constructing JSON, especially if the question might contain special characters.
if [ -n "$JQ_EXISTS" ]; then
    JSON_PAYLOAD=$(jq -n --arg q "$QUESTION_TEXT" '{"question": $q}')
else
    # Basic JSON construction (less safe if $QUESTION_TEXT has quotes or special chars)
    # For production, using jq or a more robust method is highly recommended.
    JSON_PAYLOAD="{\"question\": \"${QUESTION_TEXT//\"/\\\"}\"}" # Simple quote escaping
fi

# Make the API call using curl
# -s: silent mode (no progress meter)
# -X POST: specify POST request
# -H "Content-Type: application/json": set header
# -d "$JSON_PAYLOAD": send the JSON data
# --connect-timeout 5: timeout for connection
# --max-time 60: max time for the whole operation
RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "$JSON_PAYLOAD" \
    --connect-timeout 10 \
    --max-time 120 \
    "${PREDICT_ENDPOINT}")

# Check if curl command was successful (exit code 0)
if [ $? -ne 0 ]; then
    echo "Error: curl command failed. Could not connect to the server or other network issue."
    exit 1
fi

# Check if the response is empty (could indicate a server issue not caught by curl's exit code)
if [ -z "$RESPONSE" ]; then
    echo "Error: Received an empty response from the server."
    exit 1
fi

# Process the response
if [ -n "$JQ_EXISTS" ]; then
    # Try to parse the answer using jq
    ANSWER=$(echo "$RESPONSE" | jq -r '.answer')
    ERROR_MSG=$(echo "$RESPONSE" | jq -r '.error')

    if [ "$ANSWER" != "null" ] && [ -n "$ANSWER" ]; then
        echo "----------------------------------------"
        echo "Model's Answer:"
        echo "$ANSWER"
        echo "----------------------------------------"
    elif [ "$ERROR_MSG" != "null" ] && [ -n "$ERROR_MSG" ]; then
        echo "Error from server: $ERROR_MSG"
        # Optionally print the full raw response for debugging
        # echo "Raw server response:"
        # echo "$RESPONSE" | jq '.'
        exit 1
    else
        echo "Could not parse a clear answer or error from the server response."
        echo "Raw server response:"
        echo "$RESPONSE" | jq '.' # Pretty print the full JSON
        exit 1
    fi
else
    # Basic parsing if jq is not available (less robust)
    echo "----------------------------------------"
    echo "Raw Server Response (jq not installed):"
    echo "$RESPONSE"
    echo "----------------------------------------"
    echo "Note: Install 'jq' for better JSON parsing and formatted output."
fi