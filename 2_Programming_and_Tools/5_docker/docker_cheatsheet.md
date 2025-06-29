# Docker Cheat Sheet

## Basic Commands
- **Build an image:** `docker build -t <image_name> .`
- **List images:** `docker images`
- **Run a container:** `docker run -p <host_port>:<container_port> <image_name>`
- **List running containers:** `docker ps`
- **List all containers (including stopped):** `docker ps -a`
- **Stop a container:** `docker stop <container_id_or_name>`
- **Remove a container:** `docker rm <container_id_or_name>`
- **Remove an image:** `docker rmi <image_id_or_name>`
- **Execute a command in a running container:** `docker exec -it <container_id_or_name> <command>`
- **View container logs:** `docker logs <container_id_or_name>`

## Dockerfile Basics
```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD ["python", "app.py"]
```

## Volumes
- **Mount a host directory:** `docker run -v /host/path:/container/path <image_name>`
- **Create a named volume:** `docker volume create <volume_name>`
- **Mount a named volume:** `docker run -v <volume_name>:/container/path <image_name>`

## Networks
- **Create a network:** `docker network create <network_name>`
- **Connect a container to a network:** `docker run --network <network_name> <image_name>`

## Docker Compose
- **`docker-compose.yml` example:**
```yaml
version: '3.8'
services:
  web:
    build: .
    ports:
      - "5000:5000"
  redis:
    image: "redis:alpine"
```
- **Start services:** `docker-compose up`
- **Start services in detached mode:** `docker-compose up -d`
- **Stop services:** `docker-compose down`

## Best Practices
- Use specific image tags (e.g., `python:3.9-slim-buster`) instead of `latest`.
- Minimize the number of layers in your Dockerfile.
- Use `.dockerignore` to exclude unnecessary files.
- Keep containers ephemeral (can be stopped and destroyed, then rebuilt and replaced with a minimal setup and configuration).
