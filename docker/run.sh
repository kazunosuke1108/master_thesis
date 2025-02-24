#!/bin/bash

sudo chmod 666 /var/run/docker.sock

cd `dirname $0`

# xhost +
xhost +local:user
# xhost + 192.168.1.54
nvidia-docker &> /dev/null
if [ $nvidia_docker_installed -ne 0 ] && [ $nvidia_container_toolkit_installed -ne 2 ]; then
    echo $TAG
    echo "=========================================================="
    echo "= nvidia-docker & nvidia-container-toolkit not installed ="
    echo "=========================================================="
else
    echo "=========================" 
    echo "=nvidia docker installed="
    echo "========================="

    docker run -it \
    --device /dev/snd \
    --privileged \
    --gpus all \
    -e PULSE_SERVER=unix:/run/user/$(id -u)/pulse/native \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    --env=DISPLAY=$DISPLAY \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v /run/user/$(id -u)/pulse:/run/user/$(id -u)/pulse \
    -v "/home/${USER}/.Xauthority:/home/${USER}/.Xauthority" \
    --env="QT_X11_NO_MITSHM=1" \
    --rm \
    -v "/$(pwd)/global_ros_setting.sh:/ros_setting.sh" \
    -v "/$(pwd)/ros_workspace:/catkin_ws/" \
    -v "/$(pwd)/../master_thesis_modules:/catkin_ws/src/master_thesis_modules" \
    -v "/home/${USER}/ytlab_ros_ws/ytlab_handheld_sensoring_system/ytlab_handheld_sensoring_system_modules/database:/catkin_ws/src/database" \
    -v /etc/group:/etc/group:ro \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/localtime:/etc/localtime:ro \
    -v /media:/media \
    -v /dev:/dev \
    --net host \
    ${USER}/master_thesis
fi

# /home/ytpc2024n/ytlab_ros_ws/ytlab_handheld_sensoring_system/ytlab_handheld_sensoring_system_modules/database