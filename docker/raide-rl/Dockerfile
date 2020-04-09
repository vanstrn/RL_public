
#Loading a preexisting image, try to get as close as you can before you start building.
#FROM tensorflow/tensorflow:1.15.2-gpu-py3
FROM nvidia/cudagl:10.1-runtime

#Running an update and upgrade. This is usually required before installing anything.
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
        python3-pip \
        git

#Adding custom software and packages.
RUN pip3 install tensorflow-gpu==1.14.0 gym gym_cap gym-minigrid matplotlib

RUN alias python='python3'
RUN alias pip='pip3'

#Adding entrypoint for things that need to be executed when the image is mounted.
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
