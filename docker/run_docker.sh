#!/bin/bash

# Set the image name
image_name="mle_bomberman"

# Retrieve the image ID using the image name
image_id=$(docker images | grep $image_name | awk '{print $3}')

# Define the base directory and work directory inside the container
if [[ "$(uname)" == "Darwin" ]]; then
    # macOS specific settings
    dir="/Users/jli/MLE/bomberman_rl/"
    workdir="/Users/jli/MLE/bomberman_rl/"
    # macOS might require additional settings for X11, e.g., using XQuartz
    display_var="host.docker.internal:0"

    # Execute xhost
    xhost +

elif [[ "$(uname -s)" == "Linux" ]]; then
    # Linux specific settings
    dir="/home/jli/bomberman_rl/"
    workdir="/home/jli/bomberman_rl/"
    # For Linux, use the local host address for X11 forwarding
    display_var="$DISPLAY"

else
    echo "Unsupported system"
    exit 1
fi

# Common settings for both platforms
uid=$(id -u)  # Get the current user ID
gid=$(id -g)  # Get the current group ID

# Run the Docker image with GUI support
docker run -it --rm \
    -e DISPLAY=$display_var \
    -v $dir:$workdir \
    -w $workdir \
    -u $uid:$gid \
    --network host \
    $image_id

