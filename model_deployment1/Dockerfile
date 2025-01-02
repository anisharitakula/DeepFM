FROM python:3.10-slim

ARG MODEL_URI
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

#COPY serving/serve.py /app/serve.py
COPY . /app

# Install numpy<2 first
RUN pip install --no-cache-dir "numpy<2"

# Then the remaining installations
RUN pip install --no-cache-dir mlflow fastapi numpy pydantic torch==2.2.2+cpu -f https://download.pytorch.org/whl/torch_stable.html boto3 botocore uvicorn s3fs

RUN echo "$MODEL_URI" > model_uri.txt

EXPOSE 8000
ENV PYTHONUNBUFFERED=1
CMD ["python", "serving/serve.py"]
