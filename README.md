# A Point Cloud Registration Benchmark Platform



## Introduction

In order to test the registration performance of each point cloud registration baseline, a point cloud registration test graphical interface based on PyQt5 is developed.



## Point Cloud Registration Baselines Currently Supported

- [x] ICP
- [x] DGM
- [x] DGM-Net
- [ ] [RPM-Net](https://github.com/yewzijian/RPMNet)
- [ ] [RGM](https://github.com/fukexue/RGM)
- [ ] [PCRNet](https://github.com/vinits5/pcrnet_pytorch)



## Data Simulation

### Clean Data

You can select point cloud data files (*.hdf5/h5), which contain point clouds in the format `Point-Cloud-Num x Point-Num x 3`. Random rotation matrix and translation vector will be generated with the angles in $[-45\degree,45\degree]$  and translation in $[-1,1]$ of XYZ axis.



## Run
> python3 ./src/benchmark.py