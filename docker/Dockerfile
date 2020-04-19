ARG cuda_version=10.0
ARG cudnn_version=7

FROM nvidia/cuda:10.0-base-ubuntu16.04

#COPY c:\Users\Hevra\Documents\GitHub\model-leaf_HU\environment.yml /environment.yml
#COPY c:\Users\Hevra\Documents\GitHub\model-leaf_HU\entrypoint.sh /entrypoint.sh

RUN apt-get update && apt-get install -y \
	wget \
	vim \
	bzip2\
    libglib2.0-0\
    libsm6\
    libxext6\
    libxrender-dev

# Install Miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda.sh && \
	/bin/bash Miniconda.sh -b -p /opt/conda && \
	rm Miniconda.sh

ENV TMPDIR=/var/tmp
ENV PATH /opt/conda/bin:$PATH

#Create conda environment
#RUN conda update -n base -c defaults conda
#RUN conda env create -f environment.yml
#RUN conda init bash
#RUN echo "conda activate model-env" >> ~/.bashrc
#RUN echo "cd /workspace" >> ~/.bashrc 

## Jupyter
EXPOSE 8888
## Tensorboard
EXPOSE 6006

#ENTRYPOINT /entrypoint.sh