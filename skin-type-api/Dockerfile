FROM python:3.9-slim

WORKDIR /app

# Copy all files from your repo root to /app in container
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]



