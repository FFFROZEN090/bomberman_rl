# Build docker image for the project in Mac OS

# Define the project name
PROJECT_NAME="mle_bomberman"

# Build docker image
docker build -t $PROJECT_NAME:latest -f Dockerfile .