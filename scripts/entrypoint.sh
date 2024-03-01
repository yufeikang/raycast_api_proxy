#!/bin/sh

# check OPENAI_API_KEY env
if [ -z "$OPENAI_API_KEY" ] && [ -z "$GOOGLE_API_KEY" ]; then
    echo "OPENAI_API_KEY or GOOGLE_API_KEY must be set"
    exit 1
fi

# if has CERT_FILE env and CERT_KEY env, run uvicorn with ssl

if [ -n "$CERT_FILE" ] && [ -n "$CERT_KEY" ]; then
    echo "run uvicorn with ssl"
    python -m uvicorn app.main:app --host 0.0.0.0 --port 443 --ssl-keyfile $CERT_KEY --ssl-certfile $CERT_FILE $args $@
else
    echo "run uvicorn without ssl"
    python -m uvicorn app.main:app --host 0.0.0.0 --port 80 $@
fi
