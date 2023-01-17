FROM mambaorg/micromamba:0.13.0

RUN apt-get install -y procps

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/env.yaml

RUN apt-get update \
    && apt-get install --no-install-recommends -y procps \
    && micromamba install -y -n base -f /tmp/env.yaml \
    && micromamba clean -a && rm -rf /var/lib/{apt,dpkg,cache,log}

ENV PATH="${PATH}:/opt/conda/bin"
WORKDIR /background_subtraction
COPY . .
