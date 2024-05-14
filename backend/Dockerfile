FROM python:3.8.19

WORKDIR /app

COPY requirements.txt requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

# Install additional system dependencies for librosa
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY . .

CMD ["uvicorn", "app:app", "--host", "localhost", "--port", "8000"]
