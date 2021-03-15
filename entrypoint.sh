#!/bin/bash

wget -O model_download_link.txt $MODEL_DOWNLOAD_LINK

head -n 4 model_download_link.txt > model_download_link2.txt 
/bin/bash model_download.sh model_download_link2.txt

torchserve --start --ncs --model-store=${PWD}/model-store &
sleep 15

apt install -y curl
ls -1 model-store | while read line; do
  curl -X POST "http://localhost:8081/models?url=${line}" 
  sleep 20
done
python server.py
