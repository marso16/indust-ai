FROM python:3.8.19  

 # Set the working directory inside the container
WORKDIR /app         

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --default-timeout=100 -r requirements.txt

# Install system packages needed by librosa (audio processing)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV TF_ENABLE_ONEDNN_OPTS=0

# Copy the application code into the container
COPY . .       

# Command to run the application (replace 8000 with the desired port)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"] 