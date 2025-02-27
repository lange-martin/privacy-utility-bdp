# Balancing Privacy and Utility in Correlated Data: A Study of Bayesian Differential Privacy

This repository contains:

1. `Balancing Privacy and Utility in Correlated Data Long Version.pdf` is the extended version of the submission that contains all the formal proofs of the results presented in the paper.
2. The datasets used in the experiments.
3. The source code for the utility experiments, as well as for any related plots, is found in `utility_experiment.py`.  
4. File `plots_for_paper.py` contains code to generate the plots of the formulas represented in the paper.

### Security/Privacy Issues and Ethical Concerns

The execution of any files in this repository does not pose any extraordinary security or privacy risks. No internet connection is required.

## Basic Requirements 

A 4-core CPU and 8GB of RAM will suffice. Execution time below 10 minutes, required storage below 1 GB.

### Hardware Requirements
No special hardware requirements beyond the basic requirements.

### Python requirements
The list of required python packages can be found in `requirements.txt`.

All packages can be installed by running

> `pip install -r requirements.txt`

## Datasets

The folder `datasets` contains the two datasets used for the evaluation.

- [Galton height](https://doi.org/10.7910/DVN/T0HSJ1) is a tabular dataset that contains 897 records with height, gender, family identifier, parents heights, and number of children in their family (29KB).
- [Activity data](https://www.kaggle.com/datasets/shambhavimalik/activity-data/data) is a time series that contains the number of a steps an individual took in five minute intervals during October and November 2012. It contains a total of 17,568 intervals (343KB).

Any necessary preprocessing is executed by the experiment code.

## Experiments 

We ran three experiments, one relating to a multivariate Gaussian distribution and the Galton height data, and two relating to a Markov Chain and the activity data (one for all activity data, and one for single-day activity data).
In each experiment, we measure utility (1) by recording the upper limit of the 95% confidence interval for the absolute query error of the private mechanism, and (2) the mean absolute percentage error (MAPE) of the private mechanism compared to the non-private query.
We compare the utility when using a DP mechanism, a BDP mechanism with the general bound, and a BDP mechanism with either the Gaussian bound or the Markov chain bound.

Run the experiments by executing file `utility_experiment.py`. All resulting plots will be saved in a folder called `figures`.

Execution command:
> `python utility_experiment.py` 

We executed the experiments on both a Linux and a Windows machine.

