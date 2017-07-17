## 3D-Caffe

The 3D Caffe library with supporting the 3D operations (Conv and Deconv).

### Modification
* ND support for CuDNN engine in "Convolution" layer 
* CuDNN engine for layer "Deconvolution"
* Support for ND pooling in 'Pooling' layer
* Add randomly cropping operation in "HDF5DataLayer"

Besides, This version also includes the following modification:

* Add a weighted softmax loss layer
* Add ImageSeg dataLayer to read image and segmentation mask

### Installation

1. Clone the respository and checkout ``3D-Caffe`` branch
  ```shell
  git clone https://github.com/yulequan/3D-Caffe/
  cd 3D-Caffe
  git checkout 3D-Caffe
  ```
2. Build Caffe
  ```shell
  cp Makefile.config.example Makefile.config
  vim Makefile.config
  ##uncomment USE_CUDNN := 1 if you want to use CuDNN
  make -j8
  ```

For other installation issues, please follow the offical instructions of [Caffe](http://caffe.berkeleyvision.org/installation.html).

It has been tested successfully on Ubuntu 14.04 with a) CUDA 8.0 and CuDNN 5.0; b) CUDA 7.5 and CuDNN4.0.

### Note
- Remember to use ```git checkout 3D-Caffe```. Otherwise, you only compile the offical Caffe version.

- We use **HDF5DataLayer** to read data when doing 3D operation. You need to generate the hdf5 data from original data type. You can refer [this](https://github.com/BVLC/caffe/tree/master/matlab/hdf5creation) to generate your own h5 data using Matlab.

- You can refer the [HeartSeg] (https://github.com/yulequan/HeartSeg) project as a demo on usage of this 3D-Caffe and generating the h5 file.

### Reference code
* [U-Net code](https://lmb.informatik.uni-freiburg.de/resources/opensource/unet.en.html)
