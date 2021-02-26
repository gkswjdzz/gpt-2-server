FROM pytorch/torchserve:0.3.0-gpu

USER root
# RUN apt-get update && \
#     apt-get install -y python3-pip python3-wheel python3-dev git make cmake pkg-config && \
#     rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

ARG model_name
ENV model_name $model_name

USER model-server
WORKDIR /home/model-server

COPY . .

EXPOSE 8000 8080 8081 8082

USER root
CMD ["torchserve", "--start", "--ncs", "--model-store=/home/model-server/model-store", "&", "python", "server.py"]