# GCNv2_SLAM Setup and Usage Guide

## Environment & Dependencies

1. **OS**: Ubuntu 18.04 LTS  
2. **GPU**: NVIDIA GeForce RTX 2060  
   ```
   NVIDIA-SMI 440.118.02   Driver Version: 440.118.02   CUDA Version: 10.2
   ```
3. **CPU**: Intel(R) Xeon(R) CPU E3-1230 V2 @ 3.30GHz  
4. **CUDA & cuDNN**:  
   - CUDA 10.2  
   - cuDNN compatible with CUDA 10.2  
   - CUDA compilation tools: `release 10.2, V10.2.89`

5. **PyTorch (Do NOT use prebuilt one)**  
   - Version: 1.9.1  
   - Download:
     ```bash
     wget https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.9.1+cu102.zip
     ```

6. **ORB_SLAM2**  
   - Follow the setup guide at: [https://github.com/raulmur/ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2)

---

## Build GCNv2_SLAM

1. Use the **revised `CMakeLists.txt`**
2. Run the build script:
   ```bash
   ./build.sh
   ```

---

## Test GCN Feature Extractor

### Setup Python Environment

```bash
conda create -n gcnv2_env python=3.8 -y
conda activate gcnv2_env
conda install pytorch==1.10.2 torchvision==0.11.3 cudatoolkit=10.2 -c pytorch
pip install opencv-python matplotlib
```

### Run Accuracy Visualization

```bash
python show_accuracy.py
```

### Verify Output

- After running, look for the folder `GCN_matching`
- This folder contains the **matching point visualization results**

---

## Run GCNv2_SLAM on SLAM Dataset

### Model Preparation

- Use the modified model:  
  `model/gcn2_320x240.pt`  
  *(Model is pre-modified. If needed, you can modify it from the original.)*

### Dataset Preparation

- Download datasets such as **TUM** or **EuRoC**

### Create Association File

- Use `associate.py` to export `association.txt`:
  ```bash
  python associate.py
  ```

### Run SLAM

```bash
cd ~/GCN2

GCN_PATH=/home/lab605/lab605/GCNv2_SLAM/GCN2/gcn2_320x240.pt ./rgbd_gcn \
    /home/lab605/lab605/GCNv2_SLAM/Vocabulary/GCNvoc.bin \
    /home/lab605/lab605/GCNv2_SLAM/GCN2/TUM3.yaml \
    /home/lab605/lab605/dataset/TUM/rgbd_dataset_freiburg1_xyz \
    /home/lab605/lab605/dataset/TUM/rgbd_dataset_freiburg1_xyz/association.txt
```

---

## Evaluation

Use [`evo_ape`](https://github.com/MichaelGrupp/evo) to evaluate absolute pose error:

```bash
evo_ape tum \
    /home/lab605/lab605/dataset/TUM/rgbd_dataset_freiburg1_xyz/groundtruth.txt \
    /home/lab605/lab605/GCNv2_SLAM/GCN2/KeyFrameTrajectory.txt \
    --align --plot
```

---

## Notes

- Make sure all paths are correctly set and exist.
- It's recommended to run SLAM evaluation on a clean conda environment for reproducibility.
