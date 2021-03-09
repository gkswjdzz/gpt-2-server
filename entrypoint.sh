#!/bin/bash

wget -O model_download_link.txt $MODEL_DOWNLOAD_LINK

head -n 3 model_download_link.txt > model_download_link2.txt
/bin/bash model_download.sh model_download_link2.txt

torchserve --start --ncs --model-store=/home/model-server/model-store \
& python server.py
