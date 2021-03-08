#!/bin/bash

urldecode() {
    local url_encoded="${1//+/ }"
    printf '%b' "${url_encoded//%/\\x}"
}

if [ -z "$1"  ]; then
  exit 1;
fi

while read line; do
  baseName=$(basename $line | cut -d'?' -f1)
  decodedName=$(urldecode $baseName)
  torchModelName=$(basename $decodedName)
  
  printf '[%b] Downloading...\n' "$torchModelName"
  wget -nc -O model-store/$torchModelName $line
  printf '[%b] Download Complete\n' "$torchModelName"
done < $1

# example)
# $line => https://firebasestorage.googleapis.com/v0/b/ainize-dev-1.appspot.com/o/gpt2-models%2Fgpt2-large_seth.mar?alt=media&token=0e7afd07-d04c-4dcf-a631-f5750338d888
# baseName => gpt2-models%2Fgpt2-large_seth.mar
# decodedName => gpt2-models/gpt2-large_seth.mar
