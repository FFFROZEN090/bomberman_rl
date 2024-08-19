# Use Miniconda3 as the base image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /home/bomberman

# Update the package list, clean up the apt cache to reduce image size,
# and install necessary X11, OpenGL, and other dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libhdf5-dev \
    gcc g++ \
    x11-apps libx11-dev xorg-dev libxt-dev libxrender-dev libxext-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglu1-mesa \
    mesa-utils \
    libglapi-mesa \
    libosmesa6 \
    mesa-utils-extra && \
    rm -rf /var/lib/apt/lists/*

# Install specific Python version and other Python libraries with Conda and pip
RUN conda install python=3.9
RUN conda install scipy numpy matplotlib numba pandas
RUN conda install pytorch torchvision -c pytorch
RUN pip install scikit-learn tqdm tensorflow keras tensorboardX xgboost lightgbm pathfinding pyaml igraph ujson networkx dill pyastar2d easydict sympy pygame

# Environment variables to set the display and enable software rendering
ENV DISPLAY=host.docker.internal:0.0
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV LIBGL_DEBUG=verbose

# Copy the current directory contents into the container at /home/bomberman
COPY . .

# Set the default command to bash
CMD ["/bin/bash"]