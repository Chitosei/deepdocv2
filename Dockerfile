# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install dependencies except for torch
RUN sed -i '/torch/d' requirements.txt && pip install --no-cache-dir -r requirements.txt

# Install PyTorch separately from the official index
RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126

# Install Tesseract OCR
RUN apt-get update && apt-get install -y tesseract-ocr

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
