FROM ghcr.io/prefix-dev/pixi:latest

COPY . /app
WORKDIR /app

# Add Git LFS package repository and install
RUN apt-get update && apt-get install -y curl \
    && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get install -y git-lfs

# Initialize Git LFS
RUN git lfs install

# Pull the `pixi.lock` file from LFS
RUN git-lfs pull --include="./pixi.lock"

# Install the dependencies
RUN if [ ! -d ".pixi/envs/" ]; then pixi install --environment test-cpu; fi

# Create the shell-hook bash script to activate the environment
RUN pixi shell-hook --environment test-cpu > /shell-hook.sh

# Extend the shell-hook script to run the command passed to the container
RUN echo 'exec "$@"' >> /shell-hook.sh

# Make the shell-hook script executable
ENTRYPOINT ["/bin/bash", "/shell-hook.sh"]

# Specify the workdir of the container
WORKDIR /workspace/jaxsim
