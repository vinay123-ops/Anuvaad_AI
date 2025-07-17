FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]


#run it like: docker build -t transliteration-service .
#docker run -p 7860:7860 transliteration-service
 