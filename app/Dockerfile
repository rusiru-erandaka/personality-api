# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for some scikit-learn components)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and model
COPY ./app ./app
COPY ./model ./model
COPY run.py .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "run.py"]