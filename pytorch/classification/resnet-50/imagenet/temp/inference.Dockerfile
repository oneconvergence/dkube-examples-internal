FROM python:3.7

RUN pip install django django-sslserver pillow==5.2.0 requests numpy protobuf
RUN apt-get update && apt-get -y install protobuf-compiler
RUN pip install sk-video scipy

WORKDIR /home/inference

COPY ./inference/ /home/inference/

CMD ["python3", "manage.py", "runsslserver", "0.0.0.0:9000"]
