# Installation
Modified from [Det3D](https://github.com/poodarchu/Det3D/blob/master/INSTALLATION.md).

## Requirements
* Python 3.6 or higher
* PyTorch 1.1 or higher
* CUDA 10.0 or higher
* CMake 3.13.2 or higher

## Setup
```
git clone https://github.com/qcraftai/simtrack.git
cd simtrack
pip install -r requirements.txt
```
* Compile CUDA code for IOU3D_NMS.
``` 
cd det3d/ops/iou3d_nms
python setup.py build_ext --inplace
```

* If you want to use SyncBN, please install [APEX](https://github.com/NVIDIA/apex). 
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

* For the voxel based backbone, please install [SpConv](https://github.com/traveller59/spconv).

Note: Our voxel based model was trained and tested based on SpConv1.x and may not work with the latest version. We are working on this to solve the conflict, please stay tuned. 

* Download detection [pretrained model](https://mitprod-my.sharepoint.com/personal/tianweiy_mit_edu/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Ftianweiy%5Fmit%5Fedu%2FDocuments%2FCenterPoint%2FMVP%2Fnusc%5Fcenterpoint%5Fpp%5Ffix%5Fbn%5Fz%5Fscale%2Fepoch%5F20%2Epth)

