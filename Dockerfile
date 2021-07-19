FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && apt-get install -y curl python3.8 python3-distutils python3-pip git-all vim build-essential libopencv-dev  && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3.8 /usr/bin/python3 && ln -sf /usr/bin/python3.8 /usr/bin/python

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --no-cache-dir "uvicorn[standard]" gunicorn learning-loop-node async_generator aiofiles retry debugpy pytest-asyncio psutil icecream psutil pytest autopep8

# darknet
COPY conf.sh /tmp/
ARG CONFIG
WORKDIR /
RUN git clone https://github.com/zauberzeug/darknet_alexeyAB.git darknet && cd darknet && git checkout 211bb29e9988f6204a32cd38d0720d171135873d 
RUN cd darknet && /tmp/conf.sh $CONFIG && make clean && make


WORKDIR /app/
RUN mkdir -p /data
ADD ./app /app
ENV PYTHONPATH=/app

EXPOSE 80

CMD /app/start.sh
