# 3D Lung Nodule Detection and Reconstruction Lung Nodule Detection

Lung cancer is the leading cause of cancer death worldwide, accounting for nearly a quarter of all cancer deaths. Early detection is crucial for successful treatment, but identifying lung nodules can be challenging due to their size and location. This project aims to improve the accuracy and efficiency of lung nodule detection and lung and airway reconstruction, ultimately enhancing the diagnostic capabilities of oncologists and improving patient outcomes.

## Overview
We leverage cutting-edge computer vision and machine learning techniques to develop a pipeline for 3D reconstruction of lung anatomy, lung airways, and potential malignant cancer. The pipeline consists of several stages including image pre-processing, feature extraction, model training and evaluation, and 3D reconstruction.

## Computer Vision
Our computer vision component detects and segments lung anatomy within CT scans. This stage involves image pre-processing techniques such as image normalization, filtering, and segmentation to prepare the data for further processing.

## Machine Learning
We utilize 3D Deep Convolution Neural Networks (DCNN) trained in multi-task fashion models for nodule segmentation and false-positive reduction. These models are designed to accurately identify lung nodules, providing a solid foundation for further analysis.

## 3D Reconstruction
The 3D reconstruction module creates detailed 3D models of the lung, airways, and potential nodules, which can be analyzed for signs of malignancy based on nodule shape, size, location, and proximity to airways. This stage employs computer graphics and 3D reconstruction algorithms to create realistic models of the lung and airways.

This polyhedron demo is rendered on GPU and is real-time.

<img src="./code/figs/currentWork.gif" width="800"> 

## Getting Started
To get started with the project, please follow the installation and usage instructions provided in the documentation.

### License
This project is licensed under the MIT License.

## References
[1] [Cancer Statistics](https://www.cancer.org/healthy/cancer-causes/general-info/lifetime-probability-of-developing-or-dying-from-cancer.html)

Relevant citations for computer vision and machine learning techniques will be added soon.

## Contributor:
    Bardh Rushiti (bardhrushiti@gmail.com)
