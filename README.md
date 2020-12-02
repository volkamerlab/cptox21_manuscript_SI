cptox21_manuscript_si
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.com/REPLACE_WITH_OWNER_ACCOUNT/cptox21_manuscript_si.svg?branch=master)](https://travis-ci.com/REPLACE_WITH_OWNER_ACCOUNT/cptox21_manuscript_si)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/cptox21_manuscript_si/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/cptox21_manuscript_si/branch/master)

This repository is part of the supporting information to the manuscript in preparation:

#### Assessing the Calibration in Toxicological in Vitro Models with Conformal Prediction
Morger A., Svensson F., Arvidsson McShane S., Gauraha N., Norinder U., Spjuth O., Volkamer A.

## Table of contents

* [Objective](#objective)
* [Data and methods](#data-and-methods)
* [Usage](#usage)
* [License](#license)
* [Acknowledgement](#acknowledgement)
* [Citation](#citation)

## Objective
(Back to [Table of contents](#table-of-contents))
The provided notebooks demonstrate the main workflow to obtain the results for the manuscript on 
"Assessing the Calibration in Toxicological in Vitro Models with Conformal Prediction". 
The notebooks can be used to train aggregated conformal predictors on the Tox21Train datasets and to make predictions on 
the Tox21Score datasets with and without updated calibration sets as well as after updating the complete training set.
The calibration of the trained models for the prediction on Tox21Score can be assessed with the help of calibration plots.

Notebook `1_example_endpoint` explains the individual experiments and the calibration plots in more details while 
notebook `1_all_endpoints` shows how the experiments and evaluation can be performed for all 12 Tox21 endpoints.
It is even possible to load the results from the manuscript - instead of rerunning all experiments - and visualise the evaluation in the form of calibration plots and rmsd box plots.

For an exhaustive explanation of conformal prediction and calibration plots we refer to the manuscript.


## Data and Methods
(Back to [Table of contents](#table-of-contents))

The Tox21 datasets used in this notebook were downloaded from the National Center for Advancing Translational Sciences:
https://tripod.nih.gov/tox21/challenge/data.jsp (downloaded 29.1.2019)

The molecules were standardised as described in the paper (Data and Methods/Data, preprocessing and encoding/Dataset preprocessing)
* Remove duplicates
* Use [`standardiser`](https://github.com/flatkinson/standardiser) library (discard non-organic compounds, apply structure standardisation rules, neutralise, remove salts)
* Remove small fragments and remaining mixtures
* Remove duplicates

Signature molecular descriptors were generated using the program [`CPSign`](https://arosbio.com/cpsign/), version 0.7.14. 

## Usage
(Back to [Table of contents](#table-of-contents))

The notebooks can be used to train aggregated conformal predictors on the Tox21 endpoints. The predictions of Tox21Score 
 can be compared in different experiments with and without updated calibration sets as well as with updating the 
 complete training set.
 The notebook may be adapted to use the code for different datasets. 

### Installation

1. Get your local copy of the `cptox21_manuscript_si` repository by:
    * Downloading it as a [Zip archive](https://github.com/volkamerlab/cptox21_manuscript_si/archive/master.zip) and unzipping it, or
    * Cloning it to your computer using git

    ```
    git clone https://github.com/volkamerlab/cptox21_manuscript_si.git
    ``` 

2. Install the [Anaconda](
https://docs.anaconda.com/anaconda/install/) (large download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (lighter) distribution for clean package version management.

3. Use the package manager `conda` to create an environment (called `cptox21_si`) for the notebooks. You can either use the provided `environment.yml` with which to automatically install all required dependencies (3a) 
or start with an empty environment (3b) and install the required libraries manually (5).

a. Create a conda environment including all dependencies with 
`conda env create -f environment.yml`

b. If you prefer to build your own environment, start with 
`conda create --name cptox21_si python=3.8`
   

4. Activate the conda environment

    `conda activate cptox21_si`

5. Install packages
If you successfully created your environment from the `environment.yml` file (3a), this step 5 can be skipped. 
If you started with your own environment (3b), continue by installing the following libraries: 

    `conda install jupyter`
    
    `conda install pandas`
    
    `conda install matplotlib`
    
    `conda install -c conda-forge scikit-learn`
    
    `pip install https://github.com/morgeral/nonconformist/archive/master.zip`


## License
(Back to [Table of contents](#table-of-contents))

This work is licensed under the MIT License.

## Acknowledgement
(Back to [Table of contents](#table-of-contents))


## Citation
(Back to [Table of contents](#table-of-contents))

If you make use of the `cptox21_manuscript_SI` notebook, please cite:

```
@article{cptox21,
    author = {
        Morger Andrea, 
        Svensson Fredrik, 
        Arvidsson Mc Shane Staffan,
        Gauraha Niharika,
        Norinder Ulf,
        Spjuth Ola,
        Volkamer Andrea},
    title = {Assessing the Calibration in Toxicological in Vitro Models with Conformal Prediction},
    journal = {manuscript in preparation}
}
```



### Copyright

Copyright (c) 2020, volkamerlab

 
