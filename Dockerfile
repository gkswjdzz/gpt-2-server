#FROM pytorch/torchserve:0.3.0-gpu

#USER root

#RUN mkdir -p /home/model-server/model-store
# COPY model-store/ /home/model-server/

FROM python:3.7

COPY requirements.txt .
RUN pip install -r requirements.txt

USER model-server
WORKDIR /home/model-server

COPY . .

EXPOSE 8000

USER root

#RUN apt update && apt install -y wget vim
ENTRYPOINT MODEL_DOWNLOAD_LINK=$MODEL_DOWNLOAD_LINK /bin/bash entrypoint.sh
