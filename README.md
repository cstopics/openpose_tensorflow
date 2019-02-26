Do not forget to visit our web page: https://cstopics.github.io/cstopics/

# Pose Extraction RGB

This repository uses the [OpenpPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) MPI model from the CMU Perceptual Computing Lab, and uses the [caffe-tensorflow](https://github.com/linkfluence/caffe-tensorflow) converter to conver the network from Caffe to Tensorflow.

## Dependencies

It was tested with **Anaconda 3**, **python 3.6** and *Cuda 9.0*, installed following the isntructions in [this guide](https://cstopics.github.io/cstopics/vision/lectures/tensorflow_cuda).

In addition, install OpenCV:

``` bash
$ pip install opencv-python
```

## Download and convert the model

Download the model:

``` bash
$ cd model/
$ wget -O pose.caffemodel http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel
$ cd ..
```

Clone the model converter, and use it (it must be executed with Python 2.7, Anaconda 2 is recommended):

``` bash
$ cd model/
$ python caffe-tensorflow/convert.py pose.prototxt --caffemodel pose.caffemodel --standalone-output-path pose.pb
$ cd ..
```

## Testing the network

With Anaconda 3 again.

* Test with the sample photo:

``` bash
$ python testPhoto.py
```

When photo appears, press any key to exit.

* Test with the sample video:

``` bash
$ python testVideo.py
```

While the video is playing, press any key to exit.

### Thanks!
