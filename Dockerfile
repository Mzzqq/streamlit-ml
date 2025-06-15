# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend directory to the container
COPY backend/ .

# Create necessary directories if they don't exist
RUN mkdir -p model_files

# Change to the app directory
WORKDIR /app/app

# Run the FastAPI application with uvicorn
CMD ["fastapi", "run", "app/main.py"]
