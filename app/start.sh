#!/usr/bin/env bash

if [[ $1 = "debug" ]]; then
   python3 -m debugpy --listen 5678 /app/main.py
elif [[ $1 = "profile" ]]; then
    kernprof -l /app/main.py
else
    export MAX_WORKERS=1
    gunicorn -k "uvicorn.workers.UvicornWorker" -c /gunicorn_conf.py "main:app"
fi