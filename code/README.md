This `code` folder contains the notebooks and underlying code to demonstrate the computational experiments performed for the manuscript in preparation:  
#### Conformal Prediction and Exchangeability in Toxicological In Vitro Datasets (title tbd)
Morger A., Svensson F., Arvidsson McShane S., Gauraha N., Norinder U., Spjuth O., Volkamer A.

## Notebooks
### calupdate_example.ipynb
This notebooks demonstrates the main workflow to obtain the results for the manuscript for an example endpoint.
 Aggregated conformal predictors are trained on the Tox21 data.
 The predictions of Tox21 score are compared in different experiments with and without updated calibration sets 
 as well as with updating the complete training set. The predictions are evaluated with calibration plots.
### evaluate_all_endpoints.ipynb
n this notebook, it is shown how the experiments for the CPTox21 manuscript (train model and make predictions with 
Tox21 data, update the training and/or calibration set) are run for multiple endpoints. Furthermore the evaluation
 over all enpoints in the form of boxplots is provided.

## Python scripts
### cptox21.py
This file contains the classes and functions written to perform the experiments in the manuscript. 
### helper_functions.py
This file contains additional helper functions used for the calculations in the above notebooks.