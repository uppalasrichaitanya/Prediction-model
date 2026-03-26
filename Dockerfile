FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything
COPY . .

# Expose port
EXPOSE 10000

# Start FastAPI with uvicorn
CMD ["uvicorn", "api.main:app",
     "--host", "0.0.0.0",
     "--port", "10000"]
