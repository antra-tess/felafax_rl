# FROM google/cloud-sdk:slim
FROM gcr.io/google.com/cloudsdktool/cloud-sdk:slim
LABEL project="felafax-api-server"

# Install Python, pip, and venv
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create a virtual environment
RUN python3 -m venv /opt/venv

# Make sure we use the virtualenv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy only the app directory and key.json
COPY felafax /app/felafax
COPY key.json /app/key.json
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/key.json"

RUN gcloud auth activate-service-account --key-file=/app/key.json

# Set project ID
ENV PROJECT_ID="felafax-training"
RUN gcloud config set project ${PROJECT_ID}

# Define environment variable
ENV PORT=8000
# 30 mins gunicorn worker timeout
ENV WORKER_TIMEOUT=900

EXPOSE $PORT


CMD uvicorn felafax.server.main:app --host 0.0.0.0 --port $PORT --timeout-keep-alive $WORKER_TIMEOUT
