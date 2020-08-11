FROM ocdr/d3-datascience-tf-cpu:v1.14

RUN sudo pip3 install nltk

RUN sudo apt install nano -y

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]

