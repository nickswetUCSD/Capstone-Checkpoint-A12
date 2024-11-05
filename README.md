# Test Time Adaptation
Oftentimes, neural networks are forced to react to data from a test domain that does not reflect the data that the model was trained on.
For example, a self-driving car that's trained on imagery taken on a sunny day (source domain) will need additional "adaptation" to react to imagery from foggy days, imagery with poor lighting, or imagery with glare (distribution-shifted test domains). Introducing Test-Time Adaptation!: the notion of updating a model at test-time with nothing but the model itself and unlabeled test data.

This is our replication of test-time adaptation code for this [excellent repo](https://github.com/locuslab/tta_conjugate) using a test-time adaptation technique known as "Conjugate Pseudolabeling". Credits to Sachin Goyal*, Mingjie Sun*, Aditi Raghunanthan, J. Zico Kolter. Check out their [paper](https://arxiv.org/pdf/2207.09640) on the subject, too.

### Contents
- Two result `.csv` files, detailing results of the replicated experiment.
- One `TTA.ipynb`, which contains our exploration and running of the code.
- One `READme.md`, detailing use of the repo.

### Accessing and Storing Data
Data from our experiment can be found in `Experiment Results - Training.csv` and `Experiment Results - Testing.csv`.
Here, we describe:
- `Experiment Results - Training.csv`: The accuracy at each epoch of gradient descent for our two pseudolabeling techniques (hard and conjugate PL).
- `Experiment Results - Testing.csv`: The error for each corruption type in the CIFAR-10C dataset for two pseudolabeling techniques (hard and conjugate PL).

### Software Dependencies
This repository contains a primary jupyter notebook, `TTA.ipynb`, which lists all necessary imports at the very top of the notebook. 
Required imports include modern installments of:
- pytorch
- torchvision
- copy
- json
- collections

### Running Code
This repository contains a primary jupyter notebook, `TTA.ipynb`, which contains code that can be run to simulate the generation of hard pseudolabels for a distribution shift from CIFAR-10 to CIFAR-10C. The notebook can be run from top to bottom to perform this task, but it is **highly** recommended to use at least 1 gpu for this as it greatly speeds up the compute time. 

