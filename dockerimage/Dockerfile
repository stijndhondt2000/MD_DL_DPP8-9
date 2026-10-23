FROM pytorch/pytorch:2.14.1-cuda13.2-cudnn9-runtime
# Set the working directory to /project_antwerp, which is the default directory where the job will be executed
#  (which will be linked to your project in the jobDefinition)
WORKDIR /project_antwerp
COPY requirements.txt .

# Install packages inside the container if needed (python3-pip used as an example here)
# We do the apt-get update at the start and remove the data it fetches at the end, all in one single RUN command. 
# This keeps the docker image smaller. See https://docs.docker.com/v17.12/develop/develop-images/dockerfile_best-practices/#run
RUN apt-get update \
    && apt-get install git-all -y \
    && git config --global --add safe.directory /project_antwerp/ \
    && pip3 install --break-system-packages -r requirements.txt \
    && rm -rf /var/lib/apt/lists/*                                                                                                                           

