FROM python:3.12-slim

WORKDIR /app
COPY . /app

RUN apt update -y && apt install -y awscli
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "application.py"]

