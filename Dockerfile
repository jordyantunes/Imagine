FROM python:3.8-slim as compile

RUN apt-get update -y && apt-get install libopenmpi-dev build-essential -y

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# RUN mkdir -p /root/.mujoco/mjpro150

# RUN apt-get update & apt-get install curl -y

# RUN curl -O -J https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz

# RUN tar -xvf mujoco210-linux-x86_64.tar.gz

# RUN cp -r ./mujoco210/ /root/.mujoco/mjpro150/

RUN python -m pip install wheel

RUN python -m pip install \
    nltk==3.5 \
    matplotlib==3.4.0 \
    pandas==1.2.3 \
    tqdm==4.59.0 \
    tensorflow==2.4.1 \
    mpi4py==3.0.3 \
    numpy==1.19.5 \
    scikit_learn==0.24.1

# RUN python -m pip install baselines

RUN python -m pip install pygame==2.0.1 \
    gym 

RUN python -m pip install \
    torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

RUN python -m pip install deepdiff

FROM python:3.8-slim as build

RUN apt-get update -y && apt-get install openmpi-bin -y

COPY --from=compile /opt/venv /opt/venv

RUN useradd -ms /bin/bash localuser
USER localuser
WORKDIR /home/localuser

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/home/localuser"

RUN python -c "import nltk; nltk.download('punkt')"

COPY src src
COPY pretrained_weights pretrained_weights

ENTRYPOINT "/bin/bash"