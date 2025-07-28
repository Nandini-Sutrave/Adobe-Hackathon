FROM python:3.10

# Set working directory inside container
WORKDIR /app

# Copy all files into the container
COPY . .

# Install system dependencies for OCR and PDF
RUN apt-get update && \
    apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Default command (can be overridden)
CMD ["python", "predict_pdf.py"]
