# 3D Lung Nodule Detection and Reconstruction Lung Nodule Detection

Nearly a quarter of all cancer deaths are caused by lung cancer, which is the most common cancer worldwide. Early detection is essential for effective therapy, but because of their size and location, lung nodules can be difficult to spot. The goal of this study is to increase the precision and effectiveness of lung and airway reconstruction, lung nodule detection, and oncologists' ability to diagnose patients.

## Overview
We design a pipeline for 3D reconstruction of lung anatomy, lung airways, and possible malignant tumors using cutting-edge computer vision and machine learning approaches. Image pre-processing, feature extraction, model training and evaluation, and 3D reconstruction are some of the processes that make up the pipeline.

## Computer Vision
In CT scans, our computer vision component recognizes and categorizes the lung anatomy. In order to prepare the data for subsequent processing, this stage uses picture pre-processing techniques such image normalization, filtering, and segmentation.

## Machine Learning
For nodule segmentation and false-positive reduction, we use 3D Deep Convolution Neural Networks (DCNN) trained in multi-task manner models. These models are made to precisely locate lung nodules, offering a strong starting point for future investigation.

## 3D reconstruction
The detailed 3D models of the lung, airways, and possible nodules produced by the 3D reconstruction module can be examined for indications of malignancy based on their shape, size, position, and closeness to the airways. In this stage, accurate models of the lung and airways are created using computer graphics and 3D reconstruction algorithms.

This polyhedron demo is rendered on GPU and is real-time.

<img src="./code/figs/currentWork3.gif" width="800" > 

## Getting Started
To get started with the project, please follow the installation and usage instructions provided in the documentation.

### License
This project is licensed under the MIT License.

## References
[1] [Cancer Statistics](https://www.cancer.org/healthy/cancer-causes/general-info/lifetime-probability-of-developing-or-dying-from-cancer.html)

Relevant citations for computer vision and machine learning techniques will be added soon.

## Contributor:
    Bardh Rushiti (bardhrushiti@gmail.com)
