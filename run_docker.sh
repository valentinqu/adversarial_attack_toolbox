#!/bin/bash

# Set variables
IMAGE_NAME="adversarial-attack-toolbox"        # Docker image name
CONTAINER_NAME="container-1"  # Container name
LOCAL_DIR="adversarial_attack_toolbox"  # Local directory
CONTAINER_DIR="/app"          # Directory inside the container
PORT_MAPPING="8888:8888"      # Host port:Container port

# Check if the container already exists
if [ "$(docker ps -a -q -f name=${CONTAINER_NAME})" ]; then
    echo "Container ${CONTAINER_NAME} already exists. Starting..."
    docker start ${CONTAINER_NAME}
else
    echo "Container ${CONTAINER_NAME} does not exist. Creating..."
    docker run --rm -it -d \
        --gpus "device=6" \
        --name ${CONTAINER_NAME} \
        -v ${LOCAL_DIR}:${CONTAINER_DIR} \
        -p ${PORT_MAPPING} \
        ${IMAGE_NAME}
fi

# Ensure dependencies from requirements.txt are installed
echo "Installing dependencies from requirements.txt..."
docker exec -it ${CONTAINER_NAME} bash -c "cd ${CONTAINER_DIR} && pip install -r requirements.txt"

# Attach to the container
echo "Attaching to container ${CONTAINER_NAME}..."
docker exec -it ${CONTAINER_NAME} bash
