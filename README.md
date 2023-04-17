# Celltype Classification Benchmark

## Table of Contents

- [Celltype Classification Benchmark](#celltype-classification-benchmark)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Environment Setup](#environment-setup)
    - [Configuring Euler](#configuring-euler)

## Introduction

 This project is part of the *`Pilot implementation of annotation of single-cell RNA-seq data guided by AI`* project from the Swiss Institute of Bioinformatics (SIB), which aims at developing and implementing a machine learning approach to guide cell type classification for large datasets on *ASAP (Automated Single-cell Analysis Portal)* and *Bgee (a gene expression database of SIB)* resources. The initial focus will be on the datasets from Drosophila melanogaster species, and later we will consider including additional datasets and large-scale atlas datasets.

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

6. Run the script by running
  
```bash
bsub -n 4  -W 24:00 -o log -R "rusage[mem=4096]" python script.py
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
