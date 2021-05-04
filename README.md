# sers_analysis_app

## Table of contents
+ [General info](#general-info)
+ [Setup](#setup)
+ [User Guide](#user-guide)

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


## User Guide
This project has produced a fully functioning example of the possbilities for a tool which could greatly streamline the process of developing/organizing as well as deploying SERS technologies.

In order to become familiar with the project, we have included a thorough description of the workflows we envision this application being used for:
+ SERS substrate research and development
+ Mock field deployment and testing 

For a full overview of the application use, you may view our demonstration [here]().


### Launching The Program
This application has been built using _Dash_ a highly integrated python package enabling approachable web application development. As a result, the software will be hosted through a local address in your default web browser. Utilizing this package allowed the appraochable design of an application, while retaining access to the robust machine learning and in particular deep learning libraries for future improvements.

The local address will host the application after running:
```
$ index.py
```


### Importing New SERS Samples
When testing an SERS technology, a large amount of data is needed to generate useful models, especially for the promising future endeavors in _Deep Learning_. 
In order to manage this data, we have provided a useful tool for organizing and labeling data collected using a given SERS technology. 

This functionality is provided in the first tab for the application, titled _Import New Data_. The following steps overview a basic workflow for the tool.
1) Select Dataset
    + The intial install should include datasets generated throughout our project. These datasets are meant to allow the seperation of different sample data acquired on developed SERS technologies or methods. 
    
    ![Data selection](guide_src/dataset_selection.png)
    
    + The first step in importing new data is to select a dataset to append to, as seen in the image above
    
    ![Molecule selection](guide_src/molecule_selection.png)
    
    + Next, the application will populate a list of previous sampled molecule as shown, from which the user can indicate they are including new samples to
    
    ![Concentration selection](guide_src/concentration_selection.png)
    + After a molecule type is defined, the concentration label dialog will populate a list of previously used concentrations for that molecule as shown, the user indicates the appropriate choice
    
    + At this point, the labeling of the newly collected data is complete. The next step is to drag and drop or use the file browser to select the data to be imported.
        - Import files are expected to be TXT, CSV, or XLSX produced from a spectroscopy tool such as BWSpec
        - After successful upload, the spectra will be displayed for user confirmation
	
	
### Developing a __standard__ model

### Developing a __custom__ model

### Testing model with newly collected sample(s)
