#!/bin/bash
# Build Umbrella Transcriber Docker image

echo "Building Umbrella Transcriber Docker image..."
echo "========================================"

# Check if .env exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    echo "Please copy .env.template to .env and configure it"
    exit 1
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Build the image
docker build -t umbrella-transcriber:latest \
    --build-arg PYANNOTE_TOKEN=$PYANNOTE_TOKEN \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo "Build successful!"
    echo "To run: docker-compose -f docker-compose.prod.yml up -d"
else
    echo ""
    echo "Build failed!"
    exit 1
fi