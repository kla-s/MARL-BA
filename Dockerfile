# Use an official Python runtime as a base image

FROM tensorflow/tensorflow:1.14.0-gpu-py3
#1.9

ARG DEBIAN_FRONTEND=noninteractive
#ARG PYDEVD_USE_CYTHON=NO

RUN apt-get update -y && apt-get install -y \
      libgeos-dev \
      python3-pip \
      python3-tk 

RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install --upgrade pip
RUN pip3 install numpy==1.14.1
RUN pip3 install tensorpack==0.9.1
RUN pip3 install opencv-python==4.2.0.32
RUN pip3 install gl==0.2.36
RUN pip3 install pyglet==1.5.0
RUN pip3 install gym==0.17.1
RUN pip3 install SimpleITK==1.2.4
RUN pip3 install image==1.5.28
RUN pip3 install IPython==6.2.1
RUN pip3 install tensor2tensor==1.15.5
RUN pip3 install matplotlib==3.2.1
RUN pip3 install tensorboardX==2.0


#CMD tensorboard --logdir /MARL-BA/train_log/AAE_100000_20


WORKDIR /MARL-BA