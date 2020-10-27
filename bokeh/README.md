# Introduction into Visualization with Bokeh

## Prerequisites 
The only prerequisite to this tutorial is a working python environment with the necessary libraries installed. The requirements listed in the requirements.txt (PyPi) or the environment.yml (Conda). We assume that most of you know how to setup a python environment, however if you are not familiar with this process, we suggest you install Anaconda and setup the environment as follows:

First, make sure that you have Anaconda installed on your computer. 
To check if conda is installed, open a Terminal / PowerShell and type: 

`conda --version`

If a version is returned, you're set, else you can download it from https://www.anaconda.com/products/individual and follow this installation guide: https://docs.anaconda.com/anaconda/install/

---

**Windows Users:**
Although not recommended by Anaconda, make sure to tick the *Add Anaconda3 to my PATH environment variable*, because this way, you don't have to add it manually, or use the anaconda prompt. 

If you do not tick it, make sure to use the anaconda prompt in the next steps. (It should be found in your start menu)

---


## Installation 
(Make sure to reopen a new terminal after installation of conda)

First we have to setup the environment, this is done by installing all requirements into a new conda environment: 

```
cd <directory where bokeh_tutorial.ipynb resides>
conda env create -f environment.yml
```

Accept the installation if prompted. 

## Starting the Server via Jupyter

```
# Activate the conda environment
conda activate bokeh-tutorial

# Run the jupyter notebook
cd <directory where bokeh_tutorial.ipynb resides>
jupyter notebook
```

A new browser window should appear where you can open the bokeh_tutorial_exercise.ipynb.


## Alternative
It's also OK to user other toolchains which allow you to run a Jupyter notebook, such as Spyder, VSCode, etc. 

