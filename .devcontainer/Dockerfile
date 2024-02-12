FROM mcr.microsoft.com/devcontainers/base:ubuntu
COPY environment.yml /opt/environment.yml

# Enable by default the conda environment for all users.
# The prefix is the one used by the micromamba feature.
ENV CONDA_PREFIX_JAXSIM=/opt/conda/envs/jaxsim
RUN echo 'function activate_conda() {' >> /etc/bash.bashrc &&\
    echo '  eval "$(micromamba shell hook -s bash)"' >> /etc/bash.bashrc &&\
    echo '  micromamba activate ${CONDA_PREFIX_JAXSIM-$CONDA_PREFIX}' >> /etc/bash.bashrc &&\
    echo '}' >> /etc/bash.bashrc &&\
    echo '[[ -x $(which micromamba) && -d ${CONDA_PREFIX_JAXSIM-$CONDA_PREFIX} ]] && activate_conda' >> /etc/bash.bashrc &&\
    echo '[[ -x $(which micromamba) && ! -x $(which mamba) ]] && alias mamba="$(which micromamba)"' >> /etc/bash.bashrc &&\
    echo '[[ -x $(which micromamba) && ! -x $(which conda) ]] && alias conda="$(which micromamba)"' >> /etc/bash.bashrc

# Provide libGL.so.1 from the host OS
RUN sudo apt-get update &&\
    sudo apt-get install -y --no-install-recommends libgl1 &&\
    rm -rf /var/lib/apt/lists/*

# The Python extension in VSCode is not able to detect the interpreter installed with micromamba.
# As documented as follows, we provide a suggestion through an environment variable.
# https://code.visualstudio.com/docs/python/environments#_where-the-extension-looks-for-environments
ENV WORKON_HOME=$CONDA_PREFIX_JAXSIM

# Specify the workdir of the container
WORKDIR /workspace/jaxsim
