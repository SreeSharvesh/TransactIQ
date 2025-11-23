FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy backend code (including static frontend)
COPY backend/ /app/backend/

# Copy requirements and install Python deps
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose the FastAPI port
EXPOSE 8000

# Start the app
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]   