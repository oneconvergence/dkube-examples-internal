FROM ubuntu:18.04

RUN apt-get update 
RUN apt install -y python-dev python-pip
RUN pip install Flask==0.10.1 requests numpy pillow

COPY server/server.py ./
