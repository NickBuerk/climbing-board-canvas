FROM ubuntu:22.04 AS opencv

RUN set -xeu && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get dist-upgrade -y && \
    apt-get autoremove -y --purge && \
    apt-get -y autoclean && \
    apt-get upgrade

RUN set -xeu && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y curl git 

RUN set -xeu && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --fix-missing python3-pip

RUN set -xeu && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile=minimal
ENV PATH="${PATH}:/root/.cargo/bin"

RUN set -xeu && \
    curl --proto '=https' --tlsv1.2 -sSfL https://github.com/microsoft/onnxruntime/releases/download/v1.19.0/onnxruntime-linux-x64-gpu-1.19.0.tgz | tar -zxvf - -C /tmp
ENV ORT_DYLIB_PATH=/tmp/onnxruntime-linux-x64-gpu-1.19.0/lib/libonnxruntime.so

RUN set -xeu && \
    curl --proto '=https' --tlsv1.2 -sSfL https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -o /tmp/sam_vit_b_01ec64.pth

RUN pip3 install onnx opencv-python
RUN pip3 install git+https://github.com/facebookresearch/segment-anything.git
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

ENV LIBTORCH_USE_PYTORCH=1
ENV LIBTORCH_BYPASS_VERSION_CHECK=1

ENTRYPOINT [ "/bin/bash" ]