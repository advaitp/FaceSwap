# FaceSwap
## Overview 
### Traditional Approach
# ![1](https://github.com/advaitp/FaceSwap/blob/main/images/faceswap%20trad.png) 
A. Detection of Facial Landmarks
# ![2](https://github.com/advaitp/FaceSwap/blob/main/images/1.png) 

B. Warping or swaping face

1. Warping using Triangulation
# ![3](https://github.com/advaitp/FaceSwap/blob/main/images/2.png)

2. Warping using Thin Plate Spline
# ![4](https://github.com/advaitp/FaceSwap/blob/main/images/3.png)

# ![5](https://github.com/advaitp/FaceSwap/blob/main/images/face.png)
C. Blending
# ![6](https://github.com/advaitp/FaceSwap/blob/main/images/4.png)

D. Tracking face

### Deep Learning Approach
Implementation of a deep learning network proposed by Feng et al., in a research work titled ”Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network”. In this research, they compute a 2D representation of the face called UV position map, which records the 3D shape of a complete face in UV space, then train a simple Convolutional Neural Network to regress it from a single 2D image. Code from github provided by the authors has been used for our Face Swap pipeline.
Output of PRNet
# ![7](https://github.com/advaitp/FaceSwap/blob/main/images/5.png)

### Steps to run the code
Requirements : 
Python 3.6.0
Tensorflow tf 1.15.0


Programs : api.py demo.py demo_texture.py, test3.py, utilities.py, predictor.py and utils and Data folder in same directory.

```
pip install tensorflow-gpu==1.15
pip install cvlib
pip install dlib
pip install argparser
```

### To run the code 
```
python Wrapper.py 
```
 
### Output
Output of the program in the Results folder by default and can be changed by argparser: modename+filename eg : DelTriTest1.mp4
