FROM continuumio/miniconda
WORKDIR /usr/src/app
COPY . Background_subtraction

RUN conda env create --name bsub -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "--no-capture-output", "-n", "bsub", "/bin/bash", "-c"]

EXPOSE 5003
# The code to run when container is started:
ENTRYPOINT ["conda", "run", "-n", "bsub", "python", "background_sub.py"]
