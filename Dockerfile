# Use a base image from NVIDIA with CUDA and OpenGL support
FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04

WORKDIR /DMSR
COPY . /DMSR

# Allow to log output to the console
ENV PYTHONUNBUFFERED=1

ARG USERNAME=xiaojie
ARG USER_UID=1000
ARG USER_GID=1000

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME 

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-setuptools \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy local files into the container
RUN pip install numpy tqdm matplotlib imagecorruptions


