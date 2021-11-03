# Installation
Modified from [CenterPoint](https://github.com/tianweiy/CenterPoint/blob/master/docs/INSTALL.md)

## Requirements
* Python 3.6+
* PyTorch 1.1 or higher
* CUDA 10.0 or higher
* CMake 3.13.2 or higher

## Setup
```
git clone https://github.com/qcraftai/simtrack.git
cd simtrack
pip install -r requirements.txt
```
* Compile CUDA code for IOU3D_nms
``` 
cd det3d/ops/iou3d_nms
python setup.py build_ext --inplace
```

* If you want to use SyncBN, please install [APEX](https://github.com/NVIDIA/apex) 
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

* For VoxelNet based backbone, you need you install [spconv](https://github.com/traveller59/spconv)

Note: The voxelnet based model is trained and tested based on spconv1.x and may not work with the latest version. We are working on this, please stay tuned. 
