#!/bin/bash

torchserve --start --ncs --model-store=/home/model-server/model-store \
& python server.py \
& wget -O model_download_link.txt $MODEL_DOWNLOAD_LINK

/bin/bash model_download.sh model_download_link.txt
