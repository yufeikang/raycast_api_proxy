#!/bin/sh

# Check OPENAI_API_KEY env
if [ -z "$OPENAI_API_KEY" ]; then
    echo "OPENAI_API_KEY is not set"
    exit 1
fi

# Function to get a checksum of .py files in a directory
checksum_directory() {
    find /project/app -maxdepth 1 -type f -name "*.py" -exec md5sum {} \; | md5sum
}

# Function to start Uvicorn and capture its PID
start_uvicorn() {
    python -m uvicorn app.main:app --host 0.0.0.0 --port 80 >>"$log_file" 2>&1 &
    echo $! # Return the PID of Uvicorn process
}

# Define a log file
log_file="/project/logs/uvicorn.log"

# Ensure log directory exists
mkdir -p /project/logs

# Start Uvicorn and get its PID
uvicorn_pid=$(start_uvicorn)

# Tail the log file in the background
tail -f "$log_file" &

echo "Running checksum_directory"

# Initial checksum
last_checksum=$(checksum_directory)

# Polling for changes in /project/app
while true; do
    sleep 3 # Adjust the sleep duration as needed
    current_checksum=$(checksum_directory)
    if [ "$last_checksum" != "$current_checksum" ]; then
        echo "Changes detected in /project/app. Restarting Uvicorn..."
        kill "$uvicorn_pid"
        uvicorn_pid=$(start_uvicorn)
        last_checksum=$current_checksum
    fi
done
