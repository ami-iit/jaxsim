# syntax=docker/dockerfile:1.4
FROM mcr.microsoft.com/devcontainers/base:jammy

ARG PROJECT_NAME=jaxsim
ARG PIXI_VERSION=v0.35.0

RUN curl -o /usr/local/bin/pixi -SL https://github.com/prefix-dev/pixi/releases/download/${PIXI_VERSION}/pixi-$(uname -m)-unknown-linux-musl \
    && chmod +x /usr/local/bin/pixi \
    && pixi info

# Add LFS repository and install.
RUN apt-get update && apt-get install -y curl \
    && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt install -y git-lfs

USER vscode
WORKDIR /home/vscode

RUN echo 'eval "$(pixi completion -s bash)"' >> /home/vscode/.bashrc
