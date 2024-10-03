FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 
#nvidia/cuda:12.5.1-devel-ubuntu24.04
#nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
#nvidia/cuda:12.5.1-devel-ubuntu24.04
#nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
#nvidia/cuda:12.4.0-devel-ubuntu22.04
#nvidia/cuda:12.5.1-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update --fix-missing \
    && apt-get install -y \
    #cuda-toolkit-12-5 \
    #cuda-toolkit-12-1 \
    unzip \
    sudo \
    wget \
    libgtk2.0-dev \
    bzip2 \
    ca-certificates \
    curl \
    git \
    vim \
    g++ \
    gcc \
    graphviz \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    #libgl1-mesa-glx \ 
    #was replace by 
    libglx-mesa0 \
    xdg-utils \
    xvfb \
    libgdal-dev \
    libhdf5-dev \
    openmpi-bin \
    software-properties-common \
    ffmpeg \ 
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#Install OLLAMA ? WORKING OR NOT ?
RUN curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.3.10 sh

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -y update && \
    apt-get install -y python3.10 && \
    apt-get install -y poppler-utils

# create a user to avoid root
# ARG USER_ID
# ARG GROUP_ID
# RUN addgroup --gid $GROUP_ID user \
#     && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user \
#     && usermod -aG sudo user \
#     && echo 'user ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Switch to the user
# USER user
# WORKDIR /home/user

# Install poetry
ENV POETRY_HOME="/root/.poetry"
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# Define custom paths for virtualenvs and cache
ENV POETRY_CACHE_DIR="/root/.cache/pypoetry"
RUN poetry config cache-dir "$POETRY_CACHE_DIR"

# Set up the project directory
WORKDIR /root/llms

# Copy files required for poetry installation
COPY ./README.md ./pyproject.toml ./poetry.lock* /root/llms/

#Install pip
RUN poetry lock 
RUN poetry run pip install --upgrade pip

# Install project dependencies using poetry, excluding pystemmer for now
RUN poetry install --no-root --verbose

RUN poetry run pip install flash_attn --no-build-isolation

#define root path
ENV PYTHONPATH=$PYTHONPATH:/root/llms 
#ATTENTION C'EST POUR ATHENA  !!!!
#define root path

# ENV PYTHONPATH=$PYTHONPATH:/home/yanis.labeyrie/llms-and-rag
#ATTEENTION C'est POUR FREYja !!!!

#launch ollama server (requiered to use the LLMs) and pull the LLM Llama3 and install flash-attn
CMD ["sh", "-c", "ollama serve"]