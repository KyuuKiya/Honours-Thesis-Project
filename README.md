# Honours-Thesis-Project

## Introduction

This repository includes all related codes in the honours thesis titled "Volumetric CT Small Bowel Segmentation using Deep Learning" from the University of Sydney. 

## Dependencies

* Python == 3.7.6
* numpy == 1.19.5
* matplotlib == 3.1.3
* SimpleITK == 2.0.2
* segmentation-models == 1.0.1
* Keras == 2.4.3
* tensorflow == 2.2.0

NOTE: You do not need all dependencies with same versions to run the code. 

## Description

* 3D CT Registration: Implementation of the multi-step registration and segmentation generation pipeline based on SimpleITK
* 3D Patch Extraction: Implementation of the 3D image patch extraction process to generate the dataset
* IoU Calculation: Implementation of the IoU score calculation process to evaluate the generated segmentation
* UNet: Self implementation of 3D U-Net specific to this project based on Keras
* VNet: Self implementation of V-Net specific to this project based on Keras
* Train: The code used to load the dataset and train the model
