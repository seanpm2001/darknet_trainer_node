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

cmd=$1
cmd_args=${@:2}
case $cmd in
    b | build)
        docker kill darknet_trainer
        docker rm darknet_trainer # remove existing container
        docker build . --build-arg CONFIG=gpu-cv-cc75 -t zauberzeug/darknet-trainer-node:latest $cmd_args
        ;;
    d | debug)
        cmd_args="/app/start.sh debug"
        ;& # fall through to run line
    r | run)
        nvidia-docker run -it --memory 20g -v $(pwd)/app:/app -v $(pwd)/data:/data $run_args -e HOST=$HOST -e USERNAME=$USERNAME -e PASSWORD=$PASSWORD --rm --name darknet_trainer --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -p 8003:80 zauberzeug/darknet-trainer-node:latest $cmd_args
        ;;
    s | stop)
        docker stop darknet_trainer $cmd_args
        ;;
    k | kill)
        docker kill darknet_trainer $cmd_args
        ;;
    d | rm)
        docker kill darknet_trainer
        docker rm darknet_trainer $cmd_args
        ;;
    l | log | logs)
        docker logs -f --tail 100 $cmd_args darknet_trainer
        ;;
    e | exec)
        docker exec $cmd_args darknet_trainer
        ;;
    a | attach)
        docker exec -it $cmd_args darknet_trainer /bin/bash
        ;;
    *)
        echo "Unsupported command \"$cmd\""
        exit 1
esac

