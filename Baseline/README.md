# Readme: Baseline Models of the Project

## Folder structure
The files in this folder are structured as follows:

- models/: folder has 
  - copies of the generated models as saved to file from the notebooks. They are typically identified as `baseline_model_{model name}_{timestamp}.skl` and are pickled joblib (from SciPy) dumps. To load any of the models, make the necessary imports and `model = joblib.load(filename)`.

  - GAMs work slightly differently, they are directly pickle dumps of dictionaries. Look into the GAM-related notebooks for more details.

- <-- moving on to the notebooks -->

- baseline__forecast_test.ipynb            : this contains a test forecast using the baseline models. Tried to predict Typhoon Rai (2021), failed spectacularly.
- baseline__new_ds_tests.ipynb             : you can see the different baseline datasets being compared in one place. Includes the experimental dataset for reference, but don't look there for definitive experimental dataset/model scores.
- baseline__overview_full_ds.ipynb         : finally, the (selected) baseline models (trained on the original full baseline dataset) were tested on the test set!
- baseline__overview_downsampled.ipynb     : same as above, but the dataset is different (this one uses TSNV variant, data starting from 2008). Compare this to the experimental models for best effect.
- baseline_downsampled.ipynb               : repeating the model fitting tasks for downsampled TSNV dataset variant in one notebook.
- baseline_gp.ipynb                        : work with gaussian processes, requires GPyTorch (with GPU support).
- baseline_linear.ipynb                    : work on linear models and GAM, using the original baseline dataset. Comparable with other `baseline_linear_{blablabla}.ipynb` notebooks.
- baseline_linear_PCA.ipynb                : same as above except the approach was to generate polynomial features and then filter them with PCA. Didn't work very well, even if MARS was involved.
- baseline_linear_{nv|ts|tsnv}.ipynb       : testing whether linear models worked better with a different baseline dataset variant. Only the TSNV one had decent results; it also contains code to make a functional GAM.
- baseline_mlp.ipynb                       : work on sklearn MLPs. The execution results aren't trustworthy because I didn't let the models finish fitting. If you have 8 hours to spare, you can try, tho.
- baseline_pytorch.ipynb                   : work on PyTorch MLPs. Please help debug (perhaps the loss function is wrong?), it always has 0 accuracy.
- baseline_sktime.ipynb                    : work on sktime time series modelling tools. Note that the peculiar requirement imposed by sktime regarding multivariate input data makes cross-validation difficult.
- baseline_trees_optim.ipynb               : the place where I carried out grid search on xgboost/extra trees hyperparams.
- baseline_trees.ipynb                     : a long-assets notebook containing all the lovely work on decision tree models. dataset used was the full one, not TSNV (as in `.._optim.ipynb`). The random states were not fixed, so results may vary.
- PyTorch visualize training results.ipynb : trying to see what was wrong with PyTorch MLPs. the files it references no longer exist, just look and don't touch.

- <-- scripts -->
- baseline_gam.py                          : using this script, you can fit a GAM without manual supervision.
- baseline_linear.sh                       : use with `baseline_gam.py`. submit to HKU CS GPU farm's slurm as a batch job.
- baseline_mlp.py                          : trains a sklearn MLP or two without manual supervision. Can be left running overnight.
- baseline_mlp.sh                          : use with `baseline_mlp.py`. submit to HKU CS GPU farm's slurm as a batch job.
- baseline_pytorch_{200|500}.py            : tries to train a MLP on the GPU, requires PyTorch w/ GPU support, runs for a few hours. 
- baseline_pytorch.sh                      : use with `baseline_pytorch_{200|500}.py`. submit to HKU CS GPU farm's slurm as a batch job.
- baseline_sktime.py                       : contains the futile attempt(s) to train sktime classifiers overnight, only to realize they won't converge in 20+ hours.
- baseline_sktime.sh                       :  use with `baseline_sktime.py`. submit to HKU CS GPU farm's slurm as a batch job.

- <-- miscellaneous files -->
- earlier_models.tar.gz                    : archive containing earlier runs of the generated models for reference. Do not use, even when in doubt, they are well beyond disappointing.
- slurm-32539.out                          : result of running baseline_mlp.sh. I got my figures for the final report from this.

