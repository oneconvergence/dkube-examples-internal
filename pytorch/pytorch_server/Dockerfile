FROM ubuntu:18.04

RUN apt update && apt upgrade -y

RUN apt install python3-pip -y

COPY . .

RUN pip3 install -e .

ENTRYPOINT ["python3", "-m", "pytorchserver"]
