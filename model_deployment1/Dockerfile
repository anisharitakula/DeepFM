# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Allow build-time model URI argument
ARG MODEL_URI

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the serve.py file into the container
COPY serving/serve.py /app/serve.py

# Install any needed packages specified in requirements.txt
#RUN pip install --no-cache-dir -r requirements.txt

# Install additional serving dependencies
RUN pip install mlflow fastapi numpy pydantic torch

# Save the model URI for later use
RUN echo "$MODEL_URI" > model_uri.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run the serving script when the container launches
CMD ["python", "serve.py"]