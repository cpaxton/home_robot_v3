# Docker: Setup and Building Docker Images

This is a guide to setting up a Docker container for running the Stretch software. This is useful for running the software on a computer that doesn't have the correct dependencies installed, or for running the software in a controlled environment.

Hello Robot has a set of scripts for [building robot-side docker images](https://github.com/hello-robot/stretch_docker), which can be considered separately. Here, we go through what Docker is, why we use it, and how to build and run the **uv-based** Stretch AI image (no conda).

## What is Docker and Why Would I Use It?

Docker is a tool that allows you to run software in a container. A container is a lightweight, standalone, executable package of software that includes everything needed to run the software, including the code, runtime, system tools, libraries, and settings. Containers are isolated from each other and from the host system, so they are a great way to run software in a controlled environment.

In particular, this docker image is designed to use the correct CUDA version for the Stretch software. This is useful if you are running the software on a computer that doesn't have the correct CUDA version installed, and it makes installing the right versions of AI/ML libraries much easier.

## Installing Docker

To install Docker, follow the instructions on the [Docker website](https://docs.docker.com/get-docker/). Test by running:

```
docker run hello-world
```

### Troubleshooting

If you are having trouble installing Docker, check the [Docker documentation](https://docs.docker.com/get-docker/) for troubleshooting tips. If you are getting a permission error on Ubuntu, make sure you have added your user to the `docker` group:

```
# Make sure the group exists
sudo groupadd docker

# Add $USER to the docker group
sudo usermod -aG docker $USER

# Restart the Docker daemon so changes take effect
sudo systemctl restart docker
```

If necessary, you can apply new group changes to current session:

```
newgrp docker
```

## Building the Docker Image

The image uses **uv** and runs `install.sh -y --sim` (no conda). Ensure the segment-anything-2 submodule is present before building (the build script does this for you).

From the **root** of the repository:

```bash
# Optional: init submodule if not already done
git submodule update --init --recursive third_party/segment-anything-2

docker build -t stretch-ai_cuda-11.8:latest . -f docker/Dockerfile.cuda-11.8
```

### Use the Docker Build Script

The build script inits the submodule, reads the version from the repo, and tags the image:

```bash
./docker/build-docker.sh
# or non-interactive:
./docker/build-docker.sh -y
```

Then push to Docker Hub (see below).

### Building and Pushing to Docker Hub

This will use the Hello Robot account as an example (username: `hellorobotinc`). Login with:

```
docker login -u hellorobotinc
```

and enter a password (or create an [access token](https://hub.docker.com/settings/security)).

Then, build the image with:

```bash
docker build -t hellorobotinc/stretch-ai_cuda-11.8:latest .
docker push hellorobotinc/stretch-ai_cuda-11.8:latest
```

You can pull with:

```bash
docker pull hellorobotinc/stretch-ai_cuda-11.8:latest
```

### Building the Robot Docker Images

Similarly, there's a robot docker image build:
```bash
./docker/build-robot-docker.sh
```

This is essentially the same command:
```
docker build -t stretch-ai-ros2-bridge . -f docker/Dockerfile.ros2
```
Again, it's preferable to build using the script.

## Running the Docker Image

The image has the project's virtualenv at `/app/.venv` and **PATH is set** so `emet` and `python` work without activating. No conda.

### 1. Run a container and attach to the shell

Use `--network host` so the container can see your robot on the LAN. For GUI (e.g. rerun), allow X and set DISPLAY:

```bash
xhost si:localuser:root
docker run -it --gpus all --network host --env DISPLAY="$DISPLAY" \
    stretch-ai_cuda-11.8:latest
```

(or use `hellorobotinc/stretch-ai_cuda-11.8:latest` if you pulled from Docker Hub)

### 2. Verify container functionality

```bash
# Emet CLI
emet --help

# Torch can use GPU
python3 -c "import torch; print('cuda:', torch.cuda.is_available())"

# Run view-images demo (make sure server is running on robot)
emet app view_images --robot-ip $ROBOT_IP
# or: python3 -m emet.app.view_images --robot_ip $ROBOT_IP
```

### Tips for Windows 11

If you happen to be running on Windows 11 with WSL2, running the container with the following command will allow you to have GUI forwarded properly. ([source](https://stackoverflow.com/questions/73092750/how-to-show-gui-apps-from-docker-desktop-container-on-windows-11))

```bash
docker run -it -v /run/desktop/mnt/host/wslg/.X11-unix:/tmp/.X11-unix `
    -v /run/desktop/mnt/host/wslg:/mnt/wslg `
    -e DISPLAY=:0 `
    -e WAYLAND_DISPLAY=wayland-0 `
    -e XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir `
    -e PULSE_SERVER=/mnt/wslg/PulseServer `
    --gpus all `
    --network host `
    hellorobotinc/stretch-ai_cuda-11.8:latest
```

### Developing within Docker Container Environment

If you want to use the Docker container as a development environment and retain the changes made in the root `stretch_ai` repository, run the Docker container with the following argument to mount the cloned `stretch_ai` repository from your host filesystem to the `/app` directory inside the Docker container.

```bash
docker run -v ~/stretch_ai:/app [other_docker_options]
```

By mounting the repository this way, any changes you make to the files in the `stretch_ai` directory on your host will be immediately reflected in the `/app` directory inside the container. This allows you to see your changes live, run them and ensures they are not lost when you stop the container.
