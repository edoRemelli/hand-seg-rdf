# Hand segmentation from RGBD input

Super real-time pipeline (<2 ms/frame on CPU) for hand & wrist segmentation based on random decision forests, joint bilateral upsampling, decision adjustment and post-processing.

# Overview

The pipeline offers 3 operating modes: { TRAIN, TEST, LIVE }.

TRAIN

640x480 depth images and ground truth must be provided to train a RDF. 
Ground truth must be provided as a single channel image where hand pixels must be labeled with 255 intensity, while wrist pixel must be labeled with 60 intensity.
Note that in order to get super real-time performance the RDF is actually trained on downsampled input, post-processing is then used at run-time to increase the resolution to original value. 

TEST

Loads a trained random decision forest and evaluates the accuracy of the complete pipeline on a test set. Should be used to tune meta-parameters of the algorithm.

LIVE

Loads a trained random decision forest and runs the pipeline on streaming RGBD input acquired from Intel Realsense sensor. Note that the pipeline for now supports only such sensor and the code must be modified in order to use other sensors. 

For additional info refer to to examples/c++/h_seg.cpp, the code it’s well commented and self-explanatory. 

# Data-set and trained forest

Train-set (29820 frames), Test-set (365 frames) and the associated trained RDF can be found on:

[https://drive.google.com/open?id=0B3Uu7sqBtl6GRVlmQ2FRWW44MDg](https://drive.google.com/open?id=0B3Uu7sqBtl6GRVlmQ2FRWW44MDg)
 
# Installing and Dependencies

Fertilized Forests Decision Forest library - contained in src/

The fertilized forests library is a general purpose, platform independent, easy
to extend, Decision Forest library. You can find detailed information including
API documentation on [http://www.fertilized-forests.org](http://www.fertilized-forests.org).

Boost

OpenCV

OpenMP

Eigen



