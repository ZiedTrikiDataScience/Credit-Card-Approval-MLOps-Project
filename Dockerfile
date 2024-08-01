FROM python:3.12.4-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001

CMD ["waitress-serve" ,"--listen=0.0.0.0:5001" ,"app:app"]
#CMD ["sh", "-c", "cd /app && waitress-serve --listen=0.0.0.0:5001 app:app"]
