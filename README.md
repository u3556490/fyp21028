# HKU CompSc COMP4802 Extended Final Year Project (fyp21028)

Welcome! This project was a pain.

Technical personnel may take a look at the readme files in each of these top-level directories to get an idea of what on earth is going on. Alternatively, the included project reports can be read, but it's not recommended (too time consooming).

Non-technical people can go to [the project blog/website](https://wp.cs.hku.hk/2021/fyp21028/), which will hopefully stay afloat for a reasonably long time.

## What are included in this repo?
Pretty much every bit of code ever written for the project and all the execution outcomes (as of 18 Apr 2022). The models generated are *not* included because some of them are too unwieldy (1.89GB).

Here is a quick summary of the top-level contents:
- Baseline/     : this folder contains all the runnable stuff and loadable stuff related to the baseline models.   
- Dataset/      : this folder contains all the runnable stuff and loadable stuff related to the datasets used.  
- Experimental/ : this folder.... related to the experimental models, and contains a sample operational forecast.  
- Reports/      : contains paperwork.  
- dev_env.yml   : a dump of the development virtual environment (`conda env export`), take a look if your venv doesn't work like mine did.
- pynio.yml     : a dump of the venv in which I performed the GRIB file processing tasks, because, PyNIO conflicts with some package in the (already quite bloated) development environment.
- install.txt   : the series of commands to type into the terminal if you want to install a (reasonably) clean development environment, presumably with GPyTorch support (install it via pip); but the PyTorch thing was unused so a barebones env with sklearn should be a decent starting point. MARS (py-earth) conflicts with sktime/GPyTorch btw, so you'll have some difficulty keeping them under the same hood - it's hard to get them in one place as comfartably as Tito's Yugoslavia.

## How do I get started and run your crappy code?
Step 1: Install anaconda/miniconda.   
Step 2: Consult the readme files and the notes in each of the (numerous) notebooks, among which a good place to start will be the two notebooks under `Experimental/operational\ forecast`. 

One nasty thing is that some references and file paths quoted in the notebooks/Python scripts won't work out-of-the-box, because this repo does not mirror the structure of the folders on the compute nodes that I work on. Well, I don't think there will be any other poor soul that wants to run the codes and scrutize the thing.

## Do you have a funny joke for us?
Well, yes. Once upon a time there was a Japanese guy (or woman, doesn't matter) travelling in Germany with the famed Deutsche Bahn (for the uninitiated: the German railway system). He enjoyed it very much, except one small matter: the seats had some substances to which he was allergic. What would he say about the journey? "Ichiban!" (Itchy Bahn)

...I'll see myself out.

Email to `u3556490@connect.hku.hk` for support and complaints.