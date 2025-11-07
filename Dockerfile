FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

COPY apt/tuna.sources.list /etc/apt/sources.list

ENV TZ=Asia/Shanghai \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    apt-utils \
    software-properties-common \
    tzdata \
    locales \
    && apt-get -y upgrade

# environment related tools
RUN apt-get update && apt-get install -y \
    bash-completion \
    ca-certificates \
    curl \
    dnsutils \
    git \
    git-lfs \
    htop \
    inetutils-ping \
    inetutils-traceroute \
    iproute2 \
    jq \
    libcurl4 \
    liblzma5 \
    libproj-dev \
    net-tools \
    netcat \
    nload \
    openssl \
    sysstat \
    telnet \
    tmux \
    unzip \
    vim \
    wget \
    zip \
    zsh

# cmake
RUN wget -O cmake.sh https://tsingroc-private-binary.oss-cn-beijing.aliyuncs.com/cmake-3.26.3-linux-x86_64.sh \
    && ls \
    && chmod +x cmake.sh \
    && ./cmake.sh --skip-license --prefix=/usr/local \
    && rm cmake.sh

# protobuf+grpc for c/c++
RUN wget -O grpc.src.tar.gz http://tsingroc-private-binary.oss-cn-beijing.aliyuncs.com/grpc1.49.2.src.tar.gz \
    && tar -xzf grpc.src.tar.gz \
    && cd grpc \
    && mkdir -p "cmake/build" \
    && cd "cmake/build" \
    && cmake ../.. -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF \
    && make -j && make -j install && ldconfig && cd ../../ \
    && cd .. && rm -r grpc \
    && rm grpc.src.tar.gz

RUN apt-get update

ARG CONDA_PATH=/root/.conda
ARG CONDA=${CONDA_PATH}/bin/conda
ARG SIMULET_PATH=/root/.simulet
ARG PYTHON_VERSION=3.7

RUN wget "https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O /root/Miniconda3.sh \
    && bash /root/Miniconda3.sh -b -p $CONDA_PATH \
    && rm /root/Miniconda3.sh
COPY .condarc /root/.condarc
RUN $CONDA create -n simulet -c conda-forge python=$PYTHON_VERSION libstdcxx-ng=12
RUN chsh -s /usr/bin/zsh
COPY cuda $SIMULET_PATH
RUN $CONDA run --no-capture-output -n simulet /usr/bin/zsh -c "\
       cd $SIMULET_PATH \
    && pip install -r requirements.txt \
    && python -m grpc_tools.protoc -Iprotos --python_out=api --grpc_python_out=api protos/wolong/**/*.proto \
    && mkdir -p build \
    && cd build/ \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && make -j \
    && cp ./src/_simulet* ../python_wrapper/src/simulet/ \
    && cd .. \
    && pip install ./python_wrapper --no-cache"
COPY .zshrc /root/.zshrc
COPY omz.zip /root/omz.zip
RUN cd /root && unzip omz.zip && rm omz.zip
COPY TSC-example /root/TSC-example
CMD ["sleep", "infinity"]
