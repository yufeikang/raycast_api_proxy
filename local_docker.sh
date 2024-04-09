#!/bin/bash

docker build -t raycast .
docker run --rm -it \
    $([[ -n $OPENAI_API_KEY ]] && echo -n "-e OPENAI_API_KEY=$OPENAI_API_KEY") \
    $([[ -n $GOOGLE_API_KEY ]] && echo -n "-e GOOGLE_API_KEY=$GOOGLE_API_KEY") \
    $([[ -f .env ]] && echo -n "--env-file .env") \
    -p 443:443 \
    --dns 1.1.1.1 \
    -v $PWD/cert/:/data/cert \
    -e CERT_FILE=/data/cert/backend.raycast.com.cert.pem \
    -e CERT_KEY=/data/cert/backend.raycast.com.key.pem \
    -e LOG_LEVEL=DEBUG \
    -e ALLOWED_USERS=$ALLOWED_USERS \
    -v $PWD/app:/project/app \
    raycast --reload
