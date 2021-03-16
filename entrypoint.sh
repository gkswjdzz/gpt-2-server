#!/bin/bash
apt update && apt install -y wget vim curl
wget -O model_download_link.txt $MODEL_DOWNLOAD_LINK

head -n 10 model_download_link.txt > model_download_link2.txt 
/bin/bash model_download.sh model_download_link2.txt

torchserve --start --ncs --model-store=${PWD}/model-store &
sleep 15

# ls -1 model-store | while read line; do
#  curl -X POST "http://localhost:8081/models?url=${line}" 
#  sleep 20
# done

TOKENIZERS_PARALLELISM=0 python server.py
