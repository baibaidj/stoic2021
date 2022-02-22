ARG UBUNTU_VERSION=18.04
ARG CUDA_VERSION=11.0
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION}
# An ARG declared before a FROM is outside of a build stage,
# so it can’t be used in any instruction after a FROM
ARG USER=algorithm
ARG PASSWORD=${USER}123$
ARG PYTHON_VERSION=3.7
# To use the default value of an ARG declared before the first FROM,
# use an ARG instruction without a value inside of a build stage:
ARG CUDA_VERSION

# Install ubuntu packages         libgl1-mesa-glx libglib2.0-0 \
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
        sudo \
        locales \
        openssh-server \
        libgl1-mesa-glx libglib2.0-0 ffmpeg libsm6 libxext6 \
        vim && \
    # Remove the effect of `apt-get update`
    rm -rf /var/lib/apt/lists/* && \
    # Make the "en_US.UTF-8" locale
    localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

ENV LANG en_US.utf8

# Setup timezone
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


# Create an user for the app.
# RUN useradd --create-home --shell /bin/bash --groups sudo ${USER}
RUN groupadd -r algorithm && useradd -m --no-log-init --create-home --shell /bin/bash -r -g algorithm algorithm
RUN echo ${USER}:${PASSWORD} | chpasswd
USER ${USER}
ENV HOME /home/${USER}
WORKDIR $HOME

# Install miniconda (python)
# Referenced PyTorch's Dockerfile:
#   https://github.com/pytorch/pytorch/blob/master/docker/pytorch/Dockerfile
RUN curl -o miniconda.sh https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.11.0-Linux-x86_64.sh && \
    chmod +x miniconda.sh && \
    ./miniconda.sh -b -p conda && \
    rm miniconda.sh
  
ENV PATH $HOME/conda/bin:$PATH
RUN touch $HOME/.bashrc && \
    echo "export PATH=$HOME/conda/bin:$PATH" >> $HOME/.bashrc

# RUN conda/bin/conda install -y tensorflow-gpu==1.15.0 && \
#     conda/bin/conda install pytorch==1.7.0 cudatoolkit=11.0 && \
#     conda/bin/conda clean -ya
RUN pip install torch==1.7.0+cu110 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install https://pypi.tuna.tsinghua.edu.cn/packages/bc/72/d06017379ad4760dc58781c765376ce4ba5dcf3c08d37032eeefbccf1c51/tensorflow_gpu-1.15.0-cp37-cp37m-manylinux2010_x86_64.whl#sha256=1344a3541e19e5b5cfde1c7b71fb02cb2f593262841a0e064df033619137f609


RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

# RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm
USER root
RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

# ENV PATH="/home/algorithm/.local/bin:${PATH}"
# RUN python -m pip install --user -U pip

ARG FIND_LINKS=https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
RUN pip install mmcv-full==1.4.0 --no-cache-dir -f ${FIND_LINKS}\
  && pip install --no-cache-dir terminaltables cityscapesscripts

RUN pip install torchvision==0.8.1+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN pip install -r requirements.txt

COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm utils.py /opt/algorithm/
COPY --chown=algorithm:algorithm algorithm/ /opt/algorithm/algorithm/


# custom code
COPY --chown=algorithm:algorithm configs/ /opt/algorithm/configs/
COPY --chown=algorithm:algorithm mmdet/ /opt/algorithm/mmdet/
COPY --chown=algorithm:algorithm monai/ /opt/algorithm/monai/
COPY --chown=algorithm:algorithm scripts/ /opt/algorithm/scripts/
COPY --chown=algorithm:algorithm tools/ /opt/algorithm/tools/

ENTRYPOINT python -m process $0 $@

## ALGORITHM LABELS ##

# These labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=STOICAlgorithm