FROM pytorch/torchserve:0.3.0-gpu

USER root

RUN mkdir -p /home/model-server/model-store
# COPY model-store/ /home/model-server/

COPY requirements.txt .
RUN pip install -r requirements.txt

USER model-server
WORKDIR /home/model-server

COPY . .

EXPOSE 8000 8080 8081 8082

CMD ["torchserve", "--start", "--ncs", "--model-store=/home/model-server/model-store", "&", "python", "server.py"]
