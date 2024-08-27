# fmri-autoreg

This repo contains the code for the paper [A benchmark of individual auto-regressive models in a massive fMRI dataset](https://doi.org/10.1162/imag_a_00228).

![graphical_abstract](images/Graphical_abstract.jpg)

Several deep learning architectures are trained to predict the evolution of BOLD time series.
The code allows to perform gridsearch on each one of the architecture, to compare them and find the model architecture that is the most appropriate for BOLD time series auto-regression.

## Installation

The study was performed using Python 3.9.

Install torch 1.12.1:

`pip install torch==1.12.1`

Install the rest of the dependancies:

`pip install -r requirements.txt`

> **Note:** It is necessary to install torch prior to the torch-geometric dependencies. That is why the torch install command must be done before the one with the requirements file. 

Install this repo as a local module:

`pip install -e .`

## Dataset availability

The Neuromod datasets are available through an inter-institutional data transfer agreement. A complete description of the process to access the datasets is available at the following url: https://docs.cneuromod.ca/en/latest/ACCESS.html.

## Citation

To cite this work, please use the following bibtex entry:


>@article{10.1162/imag_a_00228,\
    author = {Paugam, François and Pinsard, Basile and Lajoie, Guillaume and Bellec, Pierre},\
    title = "{A benchmark of individual auto-regressive models in a massive fMRI dataset}",\
    journal = {Imaging Neuroscience},\
    volume = {2},\
    pages = {1-23},\
    year = {2024},\
    month = {07},\
    issn = {2837-6056},\
    doi = {10.1162/imag_a_00228},\
    url = {https://doi.org/10.1162/imag\_a\_00228},\
    eprint = {https://direct.mit.edu/imag/article-pdf/doi/10.1162/imag\_a\_00228/2461525/imag\_a\_00228.pdf},\
}

