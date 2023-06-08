FROM nvcr.io/nvidia/pytorch:22.12-py3 

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && apt-get install -y --no-install-recommends \
    locales \
    wget \
    build-essential \
    vim \
    htop \
    curl \
    git less ssh cmake \
    zip unzip gzip bzip2 \
    python3-tk gcc g++ libpq-dev

RUN apt -y install openssh-server openssh-client
# RUN pip install salesforce-lavis
# RUN pip install flask
# RUN pip install -U flask-cors
# RUN pip install gradio
RUN pip install streamlit

# COPY blip2_backend /app
RUN git clone https://github.com/dxli94/InstructBLIP-demo.git /InstructBLIP-demo
WORKDIR /InstructBLIP-demo

RUN pip install -e .

ENV PYTHONPATH="${PYTHONPATH}:./:/InstructBLIP-demo/"

# change working directory
WORKDIR /InstructBLIP-demo/projects/instructblip

# RUN chmod +x /BLIP2-demo/blip2_backend/run_app.sh
RUN chmod +x /InstructBLIP-demo/projects/instructblip/run_demo.sh