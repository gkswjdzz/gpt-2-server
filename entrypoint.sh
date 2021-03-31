curl -X GET $MODEL_DOWNLOAD_LINK -o model.list
while read line; do
  echo $line | xargs basename | sed 's/%2F/\//g' | awk -F'[///?]' -v var=$line '{ print var, $3 }' | awk -v q="'" '{print "curl "q $1 q" -o model-store/" $2}' | bash
done < model.list
cat model.list
torchserve --start --ncs --model-store=/home/model-server/model-store --models=all