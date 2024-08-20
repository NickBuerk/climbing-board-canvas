FROM ubuntu:22.04 AS opencv

RUN set -xeu && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get dist-upgrade -y && \
    apt-get autoremove -y --purge && \
    apt-get -y autoclean

RUN set -xeu && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y curl libopencv-dev clang libclang-dev git

RUN set -xeu && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile=minimal
ENV PATH="${PATH}:/root/.cargo/bin"

ENTRYPOINT [ "/bin/bash" ]