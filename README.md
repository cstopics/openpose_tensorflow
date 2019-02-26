Do not forget to visit our web page: https://cstopics.github.io/cstopics/

# pose_extraction_rgbd

## Dependencies

It was tested with **Anaconda 3**, **python 3.6** and *Cuda 9.0*, installed following the isntructions in [this guide](https://cstopics.github.io/cstopics/vision/lectures/tensorflow_cuda).

In addition, install OpenCV:

``` bash
$ pip install opencv-python
```

## Download and convert the model

Download the model:

``` bash
$ cd model
$ wget -O pose.caffemodel http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel
```