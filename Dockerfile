FROM pytorch/torchserve:0.3.0-gpu

USER root
RUN apt-get update && \
  apt-get install -y python3-pip python3-wheel python3-dev git make cmake pkg-config && \
  rm -rf /var/lib/apt/lists/*
RUN pip install wheel && \
        pip install transformers

RUN apt update
RUN apt install -y curl

USER model-server
WORKDIR /home/model-server

COPY entrypoint.sh .
EXPOSE 8000

CMD ["bash", "./entrypoint.sh"]
