#!/bin/bash
set -e
IMAGE_NAME=housing-api:local
docker build -t $IMAGE_NAME .
docker stop housing-api || true
docker rm housing-api || true
docker run -d --name housing-api -p 8000:8000 $IMAGE_NAME
echo "Running container housing-api on port 8000"
