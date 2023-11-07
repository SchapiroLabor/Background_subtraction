# Use a multi-architecture base image for your target architecture, e.g., arm64v8/ubuntu:latest for ARM64 or amd64/ubuntu:latest for x86
FROM ubuntu:latest

# Update and upgrade the package repositories
RUN apt-get update && apt-get upgrade -y

# Install OpenSlide and its development files
RUN apt-get install -y openslide-tools libopenslide-dev

# Install necessary build tools and dependencies for OpenSlide Python
RUN apt-get install -y python3-dev python3-pip build-essential

# Install libgl1-mesa-glx to provide libGL.so.1
RUN apt-get install -y libgl1-mesa-glx

# Create the missing directory and set the correct permissions
RUN mkdir -p /var/lib/apt/lists/partial

# Set the locale to avoid locale-related issues
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Determine the architecture of the system where the Docker image is being built
RUN dpkg --print-architecture | grep -q amd64 && ARCH=amd64 || ARCH=arm64

# Install Miniforge (multi-architecture version for ARM64 or x86)
RUN apt-get install -y wget

# Download and install Miniforge based on the detected architecture
RUN if [ "$ARCH" = "amd64" ]; then \
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O Miniforge.sh; \
  else \
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O Miniforge.sh; \
  fi && \
  /bin/bash Miniforge.sh -b -p /opt/miniforge && \
  rm Miniforge.sh
ENV PATH="/opt/miniforge/bin:$PATH"

# Set a working directory to the directory containing the Dockerfile
WORKDIR /background_subtraction

# Copy the contents of the directory containing the Dockerfile into the image
COPY . /background_subtraction

# Create a Conda environment and activate it
RUN conda create -n backsub python=3.9.13
SHELL ["conda", "run", "-n", "backsub", "/bin/bash", "-c"]

# Configure Conda channels
RUN conda config --add channels conda-forge  # Add Conda Forge channel
RUN conda config --add channels anaconda     # Add Anaconda channel

# Install Conda packages with specific versions
RUN conda install -y scikit-image=0.19.2 numexpr=2.8.3 tifffile=2022.8.12 scipy=1.9.3 procps-ng pandas=2.1.1 zarr=2.3.2
ENV PATH="/opt/miniforge/envs/backsub/bin:$PATH"

# Install OpenSlide Python
RUN pip install openslide-python
RUN pip install palom

