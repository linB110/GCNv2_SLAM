## Environment & Dependencies
1. Ubuntu 18.04 LTS

2. GPU: NVIDIA 2060
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.118.02   Driver Version: 440.118.02   CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce RTX 206...  On   | 00000000:01:00.0  On |                  N/A |
|  0%   45C    P8    17W / 184W |   1475MiB /  7981MiB |     23%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0       965      G   /usr/lib/xorg/Xorg                           192MiB |
|    0      1503      G   /usr/bin/gnome-shell                         192MiB |
|    0      2024      G   /usr/lib/firefox/firefox                       9MiB |
|    0     26423      C   python                                      1075MiB |
+-----------------------------------------------------------------------------+

3. CPU:
model name	: Intel(R) Xeon(R) CPU E3-1230 V2 @ 3.30GHz

4. CUDA && CuDNN
Cuda compilation tools, release 10.2, V10.2.89

5. pytorch 1.9.1 compatible with cuda 10.2(do not use pre-built one)
wget https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.9.1+cu102.zip

6. ORB_SLAM2
follow the setup [https://github.com/raulmur/ORB_SLAM2]

## Build GCNv2_SLAM
use revised vesion CMakeLists.txt and run ./build.sh

## Test GCN feature extractor
#setup
conda create -n gcnv2_env python=3.8 -y
conda install pytorch==1.10.2 torchvision==0.11.3 cudatoolkit=10.2 -c pytorch
pip install opencv-python matplotlib

#test
run show_accuracy.py 

#verify
search for folder named "GCN_matching", this is the folder which stored the matching point visulization result

## Run on SLAM dataset
#model_modify
use model/gcn2_320x240.pt  (model is already modified, or manually modified it form original model)

#dataset_preparation
download TUM, EuRoC... dataset for your reference

#create association.py
use associate.py to export association.txt

#run_SLAM
cd ~/GCN2
GCN_PATH=/home/lab605/lab605/GCNv2_SLAM/GCN2/gcn2_320x240.pt ./rgbd_gcn /home/lab605/lab605/GCNv2_SLAM/Vocabulary/GCNvoc.bin /home/lab605/lab605/GCNv2_SLAM/GCN2/TUM3.yaml /home/lab605/lab605/dataset/TUM/rgbd_dataset_freiburg1_xyz /home/lab605/lab605/dataset/TUM/rgbd_dataset_freiburg1_xyz/association.txt

#evaluation
use evo ape
evo_ape tum   /home/lab605/lab605/dataset/TUM/rgbd_dataset_freiburg1_xyz/groundtruth.txt   /home/lab605/lab605/GCNv2_SLAM/GCN2/KeyFrameTrajectory.txt   --align --plot 


