FROM mambaorg/micromamba:0.26.0
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes
RUN apt-get update && apt install -y procps g++ && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* 
ENV PATH="${PATH}:/opt/conda/bin"
WORKDIR /background_subtraction
COPY . .
