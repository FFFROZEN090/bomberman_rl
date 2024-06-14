FROM continuumio/miniconda3
WORKDIR /home/bomberman
RUN apt-get update
RUN apt-get install -y pkg-config
RUN apt-get install -y libhdf5-dev
RUN apt-get -y install gcc g++
# Install specific python version
RUN conda install python=3.9
RUN conda install scipy numpy matplotlib numba
RUN conda install pytorch torchvision -c pytorch
RUN pip install scikit-learn tqdm tensorflow keras tensorboardX xgboost lightgbm
RUN pip install pathfinding pyaml igraph ujson
RUN conda install pandas
RUN pip install networkx dill pyastar2d easydict sympy pygame
RUN pip install -U pygame --user
COPY . .
CMD /bin/bash
