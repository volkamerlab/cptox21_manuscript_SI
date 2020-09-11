cptox21_manuscript_si
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.com/REPLACE_WITH_OWNER_ACCOUNT/cptox21_manuscript_si.svg?branch=master)](https://travis-ci.com/REPLACE_WITH_OWNER_ACCOUNT/cptox21_manuscript_si)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/cptox21_manuscript_si/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/cptox21_manuscript_si/branch/master)

This repository is part of the supporting information to the manuscript in preparation:

#### Conformal Prediction and Exchangeability in Toxicological In Vitro Datasets
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



## Data and Methods
(Back to [Table of contents](#table-of-contents))



The datasets used in this notebook were downloaded from public databases:
todo: add link to Tox21 (downloaded 29.1.2019)

The molecules were standardised as described in the paper (Data and Methods/Dataset Preprocessing/Standardisation)
* Remove duplicates
* Use [`standardiser`](https://github.com/flatkinson/standardiser) library (discard non-organic compounds, apply structure standardisation rules, neutralise, remove salts)
* Remove small fragments and remaining mixtures
* Remove duplicates

Signatures descriptors were generated using `cpsign` 

## Usage
(Back to [Table of contents](#table-of-contents))

The notebook can be used to train aggregated conformal predictors on the Tox21 endpoints. The predictions of Tox21 score 
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

3. Use the package manager `conda` to create an environment (called `cptox21_si`) for the notebooks.
    
    `conda create --name cptox21_si python=3.8`

4. Activate the conda environment

    `conda activate cptox21_si`

5. Install packages

    `conda install jupyter`
    `conda install pandas`
    `conda install matplotlib`
    `conda install -c conda-forge scikit-learn`
    `pip install nonconformist`


## License
(Back to [Table of contents](#table-of-contents))

This work is licensed under the BSD 3-Clause "New" or "Revised" License.??
# fixme: which license will we use??

## Acknowledgement
(Back to [Table of contents](#table-of-contents))

* ((AM and AV would like to thank Jaime Rodríguez-Guerra for supporting the set up and reviewing this repository.))
* Uppmax
* FUBright Mobility

## Citation
(Back to [Table of contents](#table-of-contents))

If you make use of the `cptox21_manuscript_SI` notebook, please cite:

```

```



### Copyright

Copyright (c) 2020, volkamerlab

 