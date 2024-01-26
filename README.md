# Celltype Classification Benchmark with both Hierarchical and Non-hierarchical Labels

## Table of Contents

- [Celltype Classification Benchmark](#celltype-classification-benchmark)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Environment Setup](#environment-setup)
    - [Configuring Euler](#configuring-euler)
  - [Benchmark architecture](#benchmark-architecture)
  - [Run benchmark](#run-benchmark)
  - [Results](#results)

## Introduction

 This project is part of the *`Pilot implementation of annotation of single-cell RNA-seq data guided by AI`* project from the Swiss Institute of Bioinformatics (SIB), which aims at developing and implementing a machine learning approach to guide cell type classification for large datasets on *ASAP (Automated Single-cell Analysis Portal)* and *Bgee (a gene expression database of SIB)* resources. It aims at benchmarking  both single label and path (multi-label) annotation pipelines for the datasets: Bgee and ASAP resources from scRNA-seq. We compared four different preprocessing methods enclosing covariants removal and dimension reduction. Then we calibrated and compared five flat models, two local models and  two global models. A report with detailed explanation can be found [here](https://www.overleaf.com/read/fsbbnqdqxknk#dcb46d) on Overleaf.

## Environment Setup

### Configuring Euler

The project is currently being built on [Euler](https://scicomp.ethz.ch/wiki/Euler). The configuration steps are as below.

1. Log in to your server via `ssh` and `git clone` the repository to `$HOME`

2. Load the  modules by running the command

```bash
source /cluster/apps/local/env2lmod.sh
set_software_stack.sh new

module load gcc/8.2.0 
module load python/3.10.4
module load hdf5/1.10.1
```

1. Create a new virtual environment by running the command

```bash
python -m venv ~/sib_ai_benchmark/.venv
```

4. Activate the virtual environment by running the following command. This will change your shell prompt to indicate that you are now working inside the virtual environment. Run`deactivate`  to exit the virtual environment.

```bash
source ~/sib_ai_benchmark/.venv/bin/activate
```

5. Install and update required packages by running

```bash
pip3 install --upgrade -r   ~/sib_ai_benchmark/requirements.txt
```

6. Run the application by running the following command. `~/sib_ai_benchmark/src/app.py` should be replaced by your own script in case of testing the configuration.
  
```bash
bsub -n 10  -W 24:00 -o log -R "rusage[mem=2048]" python ~/sib_ai_benchmark/src/app.py
```

7. Enable the environments upon next login session and set alias by running

```bash
cat <<EOF >> ~/.bashrc

module load gcc/8.2.0 
module load python/3.10.4
module load hdf5/1.10.1

source ~/sib_ai_benchmark/.venv/bin/activate

alias prun='bsub -n 4  -W 24:00 -o log -R "rusage[mem=4096]" python'
EOF
```

8. Reload  `.bashrc`  by running the command

```bash
source ~/.bashrc
```

## Benchmark architecture

The Benchmark architecture is introduced in a report which can be found [here](https://www.overleaf.com/read/fsbbnqdqxknk#dcb46d) on Overleaf.

## Run Benchmark

### Download data

Processed data per pre-processing method as well as the hierarchical information are stored  [here](https://drive.google.com/drive/folders/1mfgreVf5l1gshCcc10JTzUZCd2E2kMlh?usp=drive_link) on google drive. `gDrive` can be used for downloading. Data should be put under the `data-raw` folder as listed below.

```
data-raw
│   ├── pca_
│   └── scanvi_bcm
...    ...
│   └── sib_cell_type_hierarchy.tsv
```


### Commands to run

The following commands are supposed to be run under the `sib_ai_benchmark` folder.

Comparing preprocessing steps on flat models. They are supposed to be run separately.

```python
python src/app.py -e scanvi_bcm -m flat
python src/app.py -e scanvi_b -m flat
python src/app.py -e scanvi_ -m flat
python src/app.py -e pca_ -m flat
```

Comparison on annotation label.
```
python src/app.py -e scanvi_bcm -m flat local global
```

 To reuse the results of flat models from the above, the commands can be simplified to avoid run flat models again.
```
python src/app.py -e scanvi_bcm -m local global
```

Comparison on path evaluation.
```
python src/app.py -e scanvi_bcm -m local global -p
```

Run a single model or multiple models with a pre-processing method.
```
python src/app.py -e scanvi_bcm -m NeuralNet
python src/app.py -e scanvi_bcm -m NeuralNet LinearSVM
```

Upon completion, the app generates a pickle object and a PDF file. The pickle object encapsulates a dictionary containing detailed benchmarking results, while the PDF file visualizes a selected metric through a plot. Additionally, a log file is provisioned upon app launch to enable real-time monitoring of results.


### Add new model or data

A new model can be added to one of the folders under the parent folder `~/sib_ai_benchmark/src/models`: `flatModels`, `globalModels` or `localModels` . It should be wrapped with a specific name Which can be referred for running. Please check existing model file for use case.

To add a new dataset, an entry can be added to the  dictionary `experiments` in `cfg.py` file in `~/sib_ai_benchmark/src/config`.The dictionary key can be referred to run the specific method with the new dataset same as the [previous section](#commands-to-run).


## Results

The analysis of the benchmarking results is organized as a report which can be found [here](https://www.overleaf.com/read/fsbbnqdqxknk#dcb46d) on Overleaf.