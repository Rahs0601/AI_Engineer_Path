
"""
This script provides a conceptual example of how to containerize a simple Python application using Docker.
It explains the steps involved in creating a Dockerfile, building a Docker image, and running a Docker container.

To run this example, you will need Docker installed on your system.

Steps:
1. Create a simple Python application (e.g., app.py).
2. Create a Dockerfile in the same directory.
3. Build the Docker image.
4. Run the Docker container.
"""

import os

def create_sample_app():
    """Creates a simple Python application file (app.py)."""
    app_content = """
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, Docker! This is a simple Flask app.'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
"""
    with open("app.py", "w") as f:
        f.write(app_content)
    print("Created app.py")

def create_requirements_file():
    """Creates a requirements.txt file for the sample app."""
    requirements_content = "flask"
    with open("requirements.txt", "w") as f:
        f.write(requirements_content)
    print("Created requirements.txt")

def create_dockerfile():
    """Creates a Dockerfile for the sample application."""
    dockerfile_content = """
# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "app.py"]
"""
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    print("Created Dockerfile")

def explain_docker_commands():
    """Explains how to build and run the Docker image."""
    print("\n--- Docker Commands ---")
    print("1. To build the Docker image (make sure you are in the directory containing Dockerfile, app.py, and requirements.txt):")
    print("   docker build -t my-flask-app .")
    print("   (This command builds an image named 'my-flask-app' from the current directory.)")
    print("\n2. To run the Docker container:")
    print("   docker run -p 5000:5000 my-flask-app")
    print("   (This command runs the 'my-flask-app' image, mapping port 5000 from the container to port 5000 on your host machine.)")
    print("\n3. After running, open your web browser and go to http://localhost:5000 to see the Flask app running.")
    print("\n4. To stop the container (in another terminal):")
    print("   docker ps (to find the container ID or name)")
    print("   docker stop <container_id_or_name>")
    print("\n5. To remove the container:")
    print("   docker rm <container_id_or_name>")
    print("\n6. To remove the image:")
    print("   docker rmi my-flask-app")

if __name__ == "__main__":
    print("This script will guide you through creating files for a Docker example.")
    print("Please run this script in a new, empty directory to avoid overwriting existing files.")

    # You would typically create these files in the directory where you want to build your Docker image.
    # For this example, we'll just print the instructions.
    # In a real scenario, you might uncomment these lines to create the files:
    # create_sample_app()
    # create_requirements_file()
    # create_dockerfile()

    explain_docker_commands()
    print("\nExample files (app.py, requirements.txt, Dockerfile) are NOT created by this script.")
    print("You need to create them manually or uncomment the creation functions in the script.")
    print("The content for these files is provided above for your reference.")
