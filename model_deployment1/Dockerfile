FROM python:3.10-slim

ARG MODEL_URI
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY serving/serve.py /app/serve.py

RUN pip install --no-cache-dir mlflow fastapi numpy pydantic torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

RUN echo "$MODEL_URI" > model_uri.txt

EXPOSE 8000
ENV PYTHONUNBUFFERED=1
CMD ["python", "serve.py"]
