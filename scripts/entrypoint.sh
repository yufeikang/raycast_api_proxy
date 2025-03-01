#!/bin/sh

# Set the base command based on DEBUG_MODE
if [ "$DEBUG_MODE" = "1" ]; then
    echo "Starting in debug mode, waiting for debugger to attach on port 5678..."
    CMD_PREFIX="python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m"
else
    CMD_PREFIX="python -m"
fi

# if has CERT_FILE env and CERT_KEY env, run uvicorn with ssl
if [ -n "$CERT_FILE" ] && [ -n "$CERT_KEY" ]; then
    echo "run uvicorn with ssl"
    $CMD_PREFIX uvicorn raycast_proxy.main:app --host 0.0.0.0 --port 443 --ssl-keyfile $CERT_KEY --ssl-certfile $CERT_FILE $args $@
else
    echo "run uvicorn without ssl"
    $CMD_PREFIX uvicorn raycast_proxy.main:app --host 0.0.0.0 --port 80 $@
fi
