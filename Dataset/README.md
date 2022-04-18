# Readme: Dataset of the Project

## Folder structure
The files in this folder are structured as follows:

- baseline_datasets/: folder includes
  - baseline_dataset_48.gz                  : a pickled Pandas DataFrame containing the baseline dataset as a 48-hour implicit time series, with features as defined inside the corresponding Python script.

  - baseline_dataset_newvars_24.gz          : a pickled Pandas DataFrame containing the baseline dataset with a different set of features than the original.

  - baseline_dataset_ts_24.gz               : a pickled Pandas DataFrame containing the baseline dataset as a 24-hour explicit time series.
 
  - baseline_dataset_tsnv_24.gz             : a pickled Pandas DataFrame containing the baseline dataset as a 24-hour explicit time series with a new set of features.

  - baseline_dataset_tsnv_2006_24.gz        : a pickled Pandas DataFrame containing the TSNV baseline dataset but **only data since 2006 (inclusive) are included**.

  - baseline_dataset_tsnv_24.gz             : a pickled Pandas DataFrame containing the TSNV baseline dataset but **only data since 2008 are included**.

  - baseline_dataset.gz                     : a pickled Pandas DataFrame containing the original baseline dataset.

  - Baseline_visualizations.ipynb           : notebook contains some visualizations and tests of the baseline datasets.

  - merge_baseline_dataset_newvars.py       : This version of the baseline dataset merger program uses a new set of features, instead of using best track directly.

  - merge_baseline_dataset_ts_nv.py         : This version of the baseline dataset merger program creates an explicit time series for each sample, using the new variable layout (see `....newvars.py`).

  - merge_baseline_dataset_ts_nv_2008.py    : This version of the baseline dataset merger program produces a TSNV dataset with data starting from 2008.

  - merge_baseline_dataset_ts.py            : This version of the baseline dataset merger program creates an explicit time series for each sample.

  - merge_baseline_dataset.py               : This is the basic version of the baseline dataset merger program.


- dynamical/: folder houses 
  - Dynamical data processing.ipynb : a *copy* of the dynamical data processing notebook which is on the GPU farm. Requires a Python 2 environment with PyNIO installed to run.

  - dynamical_data.tar.gz           : a compressed archive of the CSV files.

  - 18 csv files                    : the predictors calculated in CSV format (no header), column 0 is key (timestamp or SID_timestamp) and column 1 value.

- experimental_datasets/: folder has
  - experimental_dataset_tsnv_24.gz : a pickled Pandas DataFrame containing the experimental dataset, aka TSNV baseline with 18 dynamical predictors for each 6-hour time step.

- traditional_data/: folder contains
  - ibtracs.csv          : the IBTrACS dataset in CSV format.

  - testing_ground.ipynb : notebook contains tests for the transformed best track data.

  - best_track.npy       : pickled NumPy array containing the output of `cleanup_best_track.py`.

  - issuance.npy         : pickled NumPy array containing the output of `cleanup_issuance.py`.


- cleanup_best_track.py : Cleans up, filters, converts and prepares the best track data, which are written to ./best_track.npy.

- cleanup_issuance.py   : Downloads, converts and cleans up the HKO TC warning signals records database, which is then saved to ./issuance.npy.

- dataset_utils.py      : Contains various utilities such as I/O and type conversions.

- README.md             : `O nie, jestesmy zgubieni!`


## How to make the Python scripts and Jupyter Notebooks run

Take the following steps.

1. `conda create -n efyp` to create a new virtual environment for the bulk of the scripts and notebooks (**except dynamical/Dynamical Data Processing.ipynb**).

2. Move the contents of `baseline_datasets/` and `traditional/` out of their respective directories, so that they are all at the same level as this readme file.

3. Check the files to ensure you have all the dependencies, such as jupyter, numpy, pandas, geopy, geographiclib and bs4.

4. Now you can try running them.

5. For the dynamical data processing notebook, you will need to have the aforementioned Python 2 PyNIO environment and folders containing the needed GRIB files before it will run.


*Precaution*: watch which dataset variant is being used!