FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
# Try root requirements.txt first, fallback to dashboard requirements
COPY requirements.txt* ./
COPY dashboard/requirements.txt ./dashboard-requirements.txt

# Install Python dependencies
# Try root requirements.txt first, fallback to dashboard requirements
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; else pip install --no-cache-dir -r dashboard-requirements.txt; fi

# Copy application code
COPY dashboard/ ./dashboard/
COPY tools/ ./tools/
COPY ai/ ./ai/
COPY runner/ ./runner/
COPY configs/ ./configs/

# Create necessary directories
RUN mkdir -p dashboard/data data/logs data/runtime

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=dashboard/app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Run the application
# Use dashboard app module entry point
CMD ["python", "-m", "dashboard.app"]
