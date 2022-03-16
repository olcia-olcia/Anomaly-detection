# set base image (host OS)
FROM python:3.7-slim

#run command to create working directory
RUN mkdir /src

# set the working directory in the container
WORKDIR /src

COPY requirements.txt .
# install dependencies

RUN pip install -r requirements.txt

# command to run on container start
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--allow-root", "--ip=0.0.0.0"]