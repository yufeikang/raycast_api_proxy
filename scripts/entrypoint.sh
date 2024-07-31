#!/bin/sh

# Check if certs exist in the mounted directory
if [ ! -f /project/cert/backend.raycast.com.cert.pem ] || [ ! -f /project/cert/backend.raycast.com.key.pem ]; then
    echo "Copying generated certificates to mounted directory..."
    cp /temp_cert/cert/* /project/cert/
fi

echo "Running app"
python -m uvicorn app.main:app --host 0.0.0.0 --port 443 --ssl-keyfile /project/cert/backend.raycast.com.key.pem --ssl-certfile /project/cert/backend.raycast.com.cert.pem
