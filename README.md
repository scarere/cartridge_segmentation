# Unsupervised Firearm Cartridge Segmentation

A Project for the Centre of Forensic Sciences, Toronto, ON, Canada

## Introduction
The forensic laboratory uses a 3D microscope to capture images of fired ammunition, specifically cartridge cases. These images are then sorted by software to determine whether the cartridge cases in question came from the same or different firearms. A preparatory step for this software is to segment the images into several distinct regions, something that is currently done manually. The goal of this project is to automate the segmentation process of these images into the following classes:

- Breach Face Impression
- Aperture Shear
- Firing Pin Impression
- Firing Pin Drag

Additionally, it is also desired to automate the detection of the direction of the firing pin drag.

## Problem Defenition
This essentially represents a multiclass segmentation problem. No Images or dataset were provided as a part of this project. Although there are various publicly available datasets containing microscopic images of firearm cartridge cases, these datasets are unlabeled. To the best of my knowledge, there exists no publicly available dataset in this domain that contains labels. This therefore rules out supervised machine learning as a potential solution (without going through the effort of annotating your own dataset).

## Instructions
This repository contains various exploratory python scripts. The final results can be obtained by running *main.py*. The main script calls on various functions in *functions.py* where most of the image processing is done. The required packages to run this script are listed in *requirements.txt*. One can also view the results of running this algorithm on the two provided images in *results.ipynb*.

## Future Work
Note that there are various improvements that could be made to this project. Namely, the aperture shear is currently segmented as part of the breach face impression. Figuring out how to separate these regions from one another is an area of improvement. (I believe that this problem would be trivial with a pretrained U-NET and a labelled dataset of just a few hundred samples). Additionally, several assumptions are made in various parts of the algorithm that could cause a lack of generalization to new images that do not meet those structured assumptions. One example includes the assumption that the breach face forms a circular and closed donut-like shape within which the firing pin impression.