#!/bin/bash

# export CUDA_LAUNCH_BLOCKING=1
sudo nvidia-settings -a "[gpu:0]/GPUFanControlState=1"  # ファン制御を有効化
sudo nvidia-settings -a "[fan:0]/GPUTargetFanSpeed=90"  # 50% に設定（適宜変更）
# sudo nvidia-smi -pm 1
# sudo nvidia-smi -lmc 300,5000

# source gpu_setting.sh