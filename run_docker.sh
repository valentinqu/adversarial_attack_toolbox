#!/bin/bash

# Set variables
IMAGE_NAME="adversarial-attack-toolbox"  # Docker image name
CONTAINER_NAME="container-1"             # Container name
LOCAL_DIR="./"                           # Local directory
CONTAINER_DIR="/app"                     # Directory inside the container
PORT_MAPPING="8888:8888"                 # Host port:Container port
GPU_DEVICE="device=6"                    # GPU device

# Check if the local directory exists
if [ ! -d "$LOCAL_DIR" ]; then
    echo "Error: Local directory $LOCAL_DIR does not exist."
    exit 1
fi

# Check if the container exists
if [ "$(docker ps -a -q -f name=${CONTAINER_NAME})" ]; then
    echo "Container ${CONTAINER_NAME} already exists. Restarting..."
    docker start ${CONTAINER_NAME}
else
    echo "Container ${CONTAINER_NAME} does not exist. Creating..."
    docker run --rm -it -d \
        --gpus "$GPU_DEVICE" \
        --name "${CONTAINER_NAME}" \
        -v "$(realpath "$LOCAL_DIR")":"${CONTAINER_DIR}" \
        -p "${PORT_MAPPING}" \
        "${IMAGE_NAME}"
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create container."
        exit 1
    fi
fi

# Ensure dependencies from requirements.txt are installed
#echo "Installing dependencies from requirements.txt..."
#docker exec -it ${CONTAINER_NAME} bash -c "
    cd ${CONTAINER_DIR} && \
    if [ -f requirements.txt ]; then 
        pip install -r requirements.txt; 
    else 
        echo 'Warning: requirements.txt not found in ${CONTAINER_DIR}'; 
    fi
"

# Attach to the container
echo "Attaching to container ${CONTAINER_NAME}..."
docker exec -it ${CONTAINER_NAME} bash
