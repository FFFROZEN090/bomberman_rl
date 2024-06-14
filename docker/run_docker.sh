# Run docker mle_bomberman:latest image
image_name=mle_bomberman
image_id=$(docker images | grep $image_name | awk '{print $3}')


# Check the system
if [ "$(uname)" == "Darwin" ]; then
    dir=/Users/jli/MLE/bomberman_rl/
    workdir=/Users/jli/MLE/bomberman_rl/
    # Get the current user id
    uid=$(id -u)
    # Get the current group id
    gid=$(id -g)
    # Run the docker image
    docker run -it --rm -v $dir:$workdir -w $workdir -u $uid:$gid $image_id

elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    dir=/home/jli/MLE/bomberman_rl/
    workdir=/home/jli/MLE/bomberman_rl/
    # Get the current user id
    uid=$(id -u)
    # Get the current group id
    gid=$(id -g)
    # Run the docker image
    docker run -it --rm -v $dir:$workdir -w $workdir -u $uid:$gid $image_id

else
    echo "Unsupported system"
fi