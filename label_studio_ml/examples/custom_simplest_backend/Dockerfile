FROM python:3.8-slim

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

WORKDIR /tmp
COPY requirements.txt .

ENV PYTHONUNBUFFERED=True \
    PORT=${PORT:-9090}

RUN pip install -r requirements.txt

# Install GIT libraries
COPY ./libs/label_studio_ml /libs/label_studio_ml
RUN pip install -e /libs/label_studio_ml

WORKDIR /app

COPY *.py /app/

CMD exec gunicorn --preload --bind :$PORT --workers 1 --threads 8 --timeout 0 _wsgi:app
