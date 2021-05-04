# sers_analysis_app

## Table of contents
* [General info](#general-info)
* [Setup](#setup)
* [Usage](#usage)

## General info
This project is for EE 494 - Senior Design at the University at Buffalo. Consisting of efforts from three students to create 
an expandable and simple user interface to facilitate SERS technology research, development, and a concept of its possible deployment.

	
## Setup
To run this project, install the following dependencies:

Using a conda env:
```
$ conda create --name sers_analysis python=3.7.9
$ conda activate sers_analysis
$ conda install -c conda-forge dash
$ conda install -c conda-forge pandas
$ conda install -c conda-forge pickle5
$ conda install -c anaconda scikit-learn
```


## Usage
This project has produced a fully functioning example of the possbilities for a tool which could greatly streamline the process of developing/organizing as well as deploying SERS technologies.

In order to become familiar with the project, we have included a thorough description of the workflows we envision this application being used for:
* SERS substrate research and development
* Mock field deployment and testing 

For a full overview of the application use, you may view our demonstration [here]().

### User Guide
From these possible use cases, an understanding of the following tasks are demanded:
1) Importing New SERS Samples

	When testing an SERS technology, a large amount of data is needed to generate useful models, especially for the promising future endeavors in _Deep Learning_.
	In order to manage this data, we have provided a useful tool for organizing and labeling data collected using a given SERS technology. 
	
	
2) Developing a __standard__ model

3) Developing a __custom__ model

4) Testing model with newly collected sample(s)
