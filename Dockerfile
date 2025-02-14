# Use Python 3.11 slim as base image for smaller size
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # UV configuration
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -ms /bin/bash vscode

# Install UV for Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and set working directory
WORKDIR /workspace

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv && \
    uv pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-root

# Switch to non-root user
USER vscode