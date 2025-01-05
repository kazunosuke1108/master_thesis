# This is an auto generated Dockerfile for ros:ros-core
# generated from docker_images/create_ros_core_image.Dockerfile.em
FROM ubuntu:20.04
ARG USER=initial
ARG GROUP=initial
ARG UID=1000
ARG GID=${UID}
ARG SHELL=/bin/bash
ARG workspace=/
#FROM $base_image
RUN echo base image: ${base_image}

#######################################################################
##                            Speeding up                            ##
#######################################################################
RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list

#######################################################################
##                      install common packages                      ##
#######################################################################
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
   pkg-config \
   apt-utils \
   wget \
   curl \
   git \
   build-essential \ 
   net-tools \
   gedit \
   terminator \
   nautilus \
   software-properties-common \
   apt-transport-https \
#    libopencv-dev \
   ffmpeg \
   x264 \
   libx264-dev \
   zip \
   unzip \
   usbutils \
   sudo \
   python3-pip \
   libusb-1.0-0-dev

#######################################################################
##                           install font                            ##
#######################################################################
RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections 
RUN apt-get update && apt-get install -y ttf-mscorefonts-installer \
    ttf-ubuntu-font-family \
    msttcorefonts -qq

RUN python3 -m pip install --upgrade pip
RUN apt-get install  -y python3-ruamel.yaml
#######################################################################
##                       install nvidia docker                       ##
#######################################################################
RUN apt-get install -y --no-install-recommends \
    libxau-dev \
    libxdmcp-dev \
    libxcb1-dev \
    libxext-dev \
    libx11-dev \
    mesa-utils \
    x11-apps

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

# Required for non-glvnd setups.
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64


#######################################################################
##                            ros install                            ##
#######################################################################
RUN apt-get update && apt-get install -q -y \
    dirmngr \
    gnupg2 \
    lsb-release \
    curl


# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list
# setup keys
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# install ros packages
ENV ROS_DISTRO noetic
RUN apt-get update && apt-get install -y \
    ros-noetic-desktop-full

RUN apt-get install -y \
    python3-rosdep \
    python3-rosinstall \ 
    python3-rosinstall-generator \ 
    python3-wstool \
    build-essential 

RUN rosdep init
RUN rosdep update

# setup entrypoint
COPY ./ros_entrypoint.sh /

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]

#######################################################################
##                   install additional packages                     ##
#######################################################################
RUN apt-get install -y libopencv-dev \
   ros-noetic-map-server \
   ros-noetic-amcl \
   ros-noetic-teleop-twist-joy \
   ros-noetic-camera-calibration \
   ros-noetic-gmapping \
   ros-noetic-ros-numpy \
   ros-noetic-rgbd-launch \
   ros-noetic-openni2-camera \
   ros-noetic-openni2-launch \
   ros-noetic-urg-node \
   ros-noetic-serial \
   ros-noetic-pcl-conversions

#######################################################################
##                       install hsr packages                        ##
#######################################################################
RUN sh -c 'echo "deb [arch=amd64] https://hsr-user:jD3k4G2e@packages.hsr.io/ros/ubuntu `lsb_release -cs` main" > /etc/apt/sources.list.d/tmc.list'
RUN sh -c 'echo "deb [arch=amd64] https://hsr-user:jD3k4G2e@packages.hsr.io/tmc/ubuntu `lsb_release -cs` multiverse main" >> /etc/apt/sources.list.d/tmc.list'
RUN sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
RUN sh -c 'echo "deb https://apache.jfrog.io/artifactory/couchdb-deb/ `lsb_release -cs` main" > /etc/apt/sources.list.d/couchdb.list'
RUN wget https://hsr-user:jD3k4G2e@packages.hsr.io/tmc.key -O - | sudo apt-key add -
RUN wget https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc -O - | sudo apt-key add -
RUN wget https://couchdb.apache.org/repo/keys.asc -O - | sudo apt-key add -
RUN wget https://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
RUN sh -c 'mkdir -p /etc/apt/auth.conf.d'
RUN sh -c '/bin/echo -e "machine packages.hsr.io\nlogin hsr-user\npassword jD3k4G2e" >/etc/apt/auth.conf.d/auth.conf'
RUN apt-get update
RUN apt-get install -y ros-noetic-tmc-desktop-full

#######################################################################
##                install transformers (huggingface)                 ##
#######################################################################
RUN pip install git+https://github.com/huggingface/transformers.git
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

#######################################################################
##                    install ultralytics (YOLO)                     ##
#######################################################################
RUN pip install ultralytics

#######################################################################
##                            delete cash                            ##
#######################################################################
RUN rm -rf /var/lib/apt/lists/*

#######################################################################
##                         make sudoer user                          ##
#######################################################################

ARG USER
ARG USER_ID
RUN groupadd -g 1000 developer && \
    useradd  -g      developer -G sudo -m -u $USER_ID -s /bin/bash ${USER} && \
    echo "${USER}:${USER}" | chpasswd

RUN echo "Defaults visiblepw" >> /etc/sudoers
RUN echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers


RUN ln -s /usr/bin/python3 /usr/bin/python

#######################################################################
##                    make user accessible to usb                    ##
#######################################################################
RUN adduser ${USER} dialout
RUN adduser ${USER} tty
RUN adduser ${USER} video
RUN adduser ${USER} root



# USER root
USER ${USER}
RUN echo "export PS1='\[\e[1;31;40m\]CLIP\[\e[0m\] \u:\w\$ '">> ~/.bashrc
RUN echo "source /ros_setting.sh">> ~/.bashrc

WORKDIR ${workspace}