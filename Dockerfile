FROM python:3.9

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download the Google Drive file
# RUN wget -O data/GoogleNews-vectors-negative300.bin.gz 'https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM'

CMD ["python", "app.py"]
