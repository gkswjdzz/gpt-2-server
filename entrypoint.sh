curl -X GET $MODEL_DOWNLOAD_LINK -o model.list
while read line; do
  echo $line | xargs basename | sed 's/%2F/\//g' | awk -F'[///?]' -v var=$line '{ print var, $3 }' | awk -v q="'" '{print "curl "q $1 q" -o model-store/" $2}' | bash
done < model.list
cat model.list

torchserve --start --ncs --model-store=/home/model-server/model-store
sleep 30
ls -1 model-store | while read line; do
  curl -X POST "http://localhost:8081/models?initial_workers=1&synchronous=true&url=${line}"
  sleep 20
done
