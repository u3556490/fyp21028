# Readme: Experimental Models of the Project

## Folder structure
The files in this folder are structured as follows:

- models/ : folder contains all the generated models, named in the format `experimental_model_{type}_{time stamp}.pkl`.  
  - ensemble and GAM were direct pickle dumps of lists/dictionaries.  
  - the others were joblib-dumped sklearn models.

- operational\ forecast/ : folder is an example of how an operational forecast could be conducted using this fOrEcAsTiNg pRoDuCt. The notebooks inside contain instructions about how to do stuff and get a forecast in around 30-45 minutes' time.

- <-- the notebooks -->
- __ensemble.ipynb             : the ensemble was fitted (on dev set) and tested (on test set) here.
- experimental__overview.ipynb : the performances of the experimental models (not the ensemble!) were gauged here using the test set.
- experimental_gp.ipynb        : the futile attempt to make an exact gaussian process estimator is here, requires sklearn (not gpytorch).
- experimental_linear.ipynb    : the linear models and GAMs are here, comparable to `baseline_linear_tsnv.ipynb`.
- experimental_mlp.ipynb       : all work on MLPs are here, except those done overnight using slurm jobs, which constitute the majority (haha). Numbers shown in the cells may not be accurate because I stop the fits prematurely when I run out of patience.
- experimental_sktime.ipynb    : time series modelling work.
- experimental_trees.ipynb     : work on decision trees. grid search and calibration carried out in the same place.

- <-- scripts and miscellaneous files -->
- experimental_mlp.py          : fits/grid searches MLPs overnight. Adjust to taste and run with `experimental_mlp.sh`.
- experimental_mlp.sh          : submits to the HKU CS GPU Farm as a batch slurm job.
- slurm-32588.out              : output of the MLP-fitting script.