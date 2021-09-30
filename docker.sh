#!/usr/bin/env bash

if [ $# -eq 0 ]
then
    echo "Usage:"
    echo
    echo "  `basename $0` (b | build)        Build"
    echo "  `basename $0` (d | debug)        Run with debugger"
    echo "  `basename $0` (r | run)          Run"
    echo "  `basename $0` (d | rm)           Remove Container"
    echo "  `basename $0` (s | stop)         Stop"
    echo "  `basename $0` (k | kill)         Kill"
    echo "  `basename $0` rm                 Remove"
    echo
    echo "  `basename $0` (l | log)                 Show log tail (last 100 lines)"
    echo "  `basename $0` (e | exec)     <command>  Execute command"
    echo "  `basename $0` (a | attach)              Attach to container with shell"
    echo
    echo "Arguments:"
    echo
    echo "  command       Command to be executed inside a container"
    exit
fi

# sourcing .env file to get configuration (see README.md)
. .env || echo "you should provide an .env file with USERNAME and PASSWORD for the Learning Loop"

name="darknet_trainer"

args="-it --rm" 
args+=" -v $(pwd)/app:/app"
args+=" -v $HOME/data:/data"
#args+=" -v $(pwd)/../learning_loop_node/learning_loop_node:/usr/local/lib/python3.8/dist-packages/learning_loop_node"
args+=" -h $HOSTNAME"
args+=" -e HOST=$HOST"
args+=" -e USERNAME=$USERNAME -e PASSWORD=$PASSWORD"
args+=" --name $name"
args+=" --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all"
args+=" -p 8003:80"

image="zauberzeug/darknet-trainer-node:latest"

cmd=$1
cmd_args=${@:2}
case $cmd in
    b | build)
        docker kill $name
        docker rm $name # remove existing container
        docker build . --build-arg CONFIG=gpu-cv-cc75 -t $image
        ;;
    d | debug)
        nvidia-docker run $args $image /app/start.sh debug
        ;;
    r | run)
        nvidia-docker run $args $cmd_args $image
        ;;
    s | stop)
        docker stop $name $cmd_args
        ;;
    k | kill)
        docker kill $name $cmd_args
        ;;
    d | rm)
        docker kill $name
        docker rm $name $cmd_args
        ;;
    l | log | logs)
        docker logs -f --tail 100 $cmd_args $name
        ;;
    e | exec)
        docker exec $name $cmd_args 
        ;;
    a | attach)
        docker exec -it $cmd_args darknet_trainer /bin/bash
        ;;
    *)
        echo "Unsupported command \"$cmd\""
        exit 1
esac

