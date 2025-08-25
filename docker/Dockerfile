# docker/Dockerfile
FROM python:3.9-slim

# Install system dependencies for OpenCV, spaCy, etc.
RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    libssl-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install OpenGL libraries for OpenCV
RUN apt-get update && \
    apt-get install -y libgl1-mesa-dev libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install ADB into container
RUN apt-get update && apt-get install -y \
    adb \
    # plus any other dependencies
    && rm -rf /var/lib/apt/lists/*

# Make work directoy
WORKDIR /app
COPY app/requirements.txt requirements.txt

COPY .env .env

# Install compatible numpy and other dependencies from requirements.txt
RUN pip install --no-cache-dir numpy==1.21.6
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/

# Download spaCy English model
RUN python -m spacy download en_core_web_sm

EXPOSE 8000

CMD ["bash", "-c", "adb start-server && python main.py"]