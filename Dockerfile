FROM continuumio/miniconda:latest

WORKDIR /home/skill-monitoring

COPY ./ ./

RUN chmod +x boot.sh

RUN conda env create -f environment.yml

RUN echo "source activate skill-monitoring" > ~/.bashrc

ENV PATH /opt/conda/envs/skill-monitoring/bin:$PATH
ENV CORE_HOST 'https://localhost:9000/v1'
ENV PYTHON_ENV 'production'

EXPOSE 5000

CMD ["python", "camera.py"]

# docker run --device=/dev/video0:/dev/video0 --name skill-monitoring --rm skill-monitoring:latest