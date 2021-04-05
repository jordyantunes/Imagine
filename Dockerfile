FROM python:3.6-slim as compile

RUN apt-get update -y && apt-get install libopenmpi-dev build-essential -y

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

FROM python:3.6-slim as build

RUN apt-get update -y && apt-get install openmpi-bin -y

COPY --from=compile /opt/venv /opt/venv
COPY src src
COPY pretrained_weights pretrained_weights

ENV PATH="/opt/venv/bin:$PATH"

RUN python -c "import nltk; nltk.download('punkt')"

ENTRYPOINT "/bin/bash"