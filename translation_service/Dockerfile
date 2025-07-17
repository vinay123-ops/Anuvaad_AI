FROM python:3.10-slim

# Install basic dependencies
RUN apt-get update && apt-get install -y git build-essential

# Install Python dependencies
COPY app/requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy app code
COPY app/ /app/

# Download NLTK data
RUN python3 -c "import nltk; nltk.download('punkt')"

# Expose Gradio default port
EXPOSE 7860

CMD ["python", "main.py"]


#run using this: 
#cd translation_service
#docker build -t translation-service .
#docker run --gpus all -p 7860:7860 translation-service
