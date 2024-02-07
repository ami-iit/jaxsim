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
