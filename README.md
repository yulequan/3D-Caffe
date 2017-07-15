## 3D-Caffe

We slightly modify the original Caffe library to supporting the 3D operations. 

The main modification includes:
* ND support for CuDNN engine in "Convolution" layer 
* CuDNN engine for layer "Deconvolution"
* Support for ND pooling in 'Pooling' layer
* Add randomly cropping operation in "HDF5DataLayer"

Besides, This version also includes the following modification:

* Add a weighted softmax loss layer
* Add ImageSeg dataLayer to read image and segmentation mask


For installation, please follow the offical instructions of [Caffe](http://caffe.berkeleyvision.org/installation.html).

It has been tested successfully on Ubuntu 14.04 with CUDA 8.0 and CuDNN 5.0.

## Note
- We use **HDF5DataLayer** to read data when we do 3D operation. You need to generate the hdf5 data from original data type. You can refer [this](https://github.com/BVLC/caffe/tree/master/matlab/hdf5creation) to generate your own h5 data using Matlab.
