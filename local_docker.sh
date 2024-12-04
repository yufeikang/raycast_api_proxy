#!/bin/bash

docker build -t raycast .
docker run --rm -it \
    $([[ -n $OPENAI_API_KEY ]] && echo -n "-e OPENAI_API_KEY=$OPENAI_API_KEY") \
    $([[ -n $GOOGLE_API_KEY ]] && echo -n "-e GOOGLE_API_KEY=$GOOGLE_API_KEY") \
    $([[ -n $ANTHROPIC_API_KEY ]] && echo -n "-e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY") \
    $([[ -f .env ]] && echo -n "--env-file .env") \
    -p 4444:443 \
    --dns 1.1.1.1 \
    -v "$PWD"/cert/:/project/cert \
    -e LOG_LEVEL=DEBUG \
    -e ALLOWED_USERS="$ALLOWED_USERS" \
    -v "$PWD"/app:/project/app \
    raycast
