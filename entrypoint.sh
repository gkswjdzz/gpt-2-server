#!/bin/bash

wget -O model_download_link.txt $MODEL_DOWNLOAD_LINK

/bin/bash model_download.sh model_download_link.txt

torchserve --start --ncs --model-store=${PWD}/model-store &
sleep 15

apt install -y curl
ls -1 model-store | while read line; do
  curl -X POST "http://localhost:8081/models?url=${line}" 
  sleep 20
done
python server.py
