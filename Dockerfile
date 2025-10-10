FROM mambaorg/micromamba:1.5.10-noble

# Copy conda environment file
COPY --chown=$MAMBA_USER:$MAMBA_USER ./environment.yml /tmp/conda.yml

# Install environment
RUN micromamba install -y -n base -f /tmp/conda.yml \
    && micromamba install -y -n base conda-forge::procps-ng \
    && micromamba env export --name base --explicit > environment.lock \
    && echo ">> CONDA_LOCK_START" \
    && cat environment.lock \
    && echo "<< CONDA_LOCK_END" \
    && micromamba clean -a -y

# Switch to root to copy everything
USER root

# Ensure micromamba binaries are in PATH
ENV PATH="$MAMBA_ROOT_PREFIX/bin:$PATH"

# Copy the rest of the current directory into /app inside the container
WORKDIR /app
COPY ./backsub .