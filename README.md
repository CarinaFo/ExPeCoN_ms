# Expectation Perception Confidence Neural Correlates (`ExPeCoN`)

    Author(s): Carina Forster et al. (in prep)
    Contact: forster@cbs.mpg.de
    Status: code has been formally reviewed
    Last update: 11.06.2024
***

## Open ToDos before

- [ ] review functionality

## Description

Scripts for the manuscript Expectation Perception Confidence Neural Correlates (`ExPeCoN`).

The `ExPeCoN` study investigates stimulus probabilities and the influence on perception and confidence in a
near-threshold somatosensory detection task in two datasets

## Project structure

Analysis code can be found in `code/`.
Code for conducting the experiment and data acquisition is in `experimental_code/`.


The project structure is based on [`scilaunch`](https://github.com/SHEscher/scilaunch).

### Data

* data will be uploaded to [OSF](https://osf.io) after publication

## Install research code as package

First, clone the project to a local folder:

```shell
git clone https://github.com/CarinaFo/ExPeCoN_ms.git
cd expecon_ms
```

Create a conda environment specific to `expecon_ms`.
Make sure you are in the local folder where you cloned the repository into:

```shell
conda create -n expecon_3.9 python=3.9.7
```

And activate the conda environment :

```shell
conda activate expecon_3.9
```

Install the project code `expecon_ms` as Python package:

```shell
pip install -e .
```

**Note**: The `-e` flag installs the package in editable mode,
i.e., changes to the code will be directly reflected in the installed package.
Moreover, the code keeps its access to the research data in the underlying folder structure.
Thus, the `-e` flag is recommended to use.


Now, add the conda environment as kernel to jupyter (notebook)

```shell
python -m ipykernel install --user --name=expecon_3.9
```

*R*-scripts of the analysis are stored in `./code/Rscripts/`.

### Explore data and analysis

Use jupyter notebooks to explore the data and analysis:

### Open the code in your preferred editor, but make sure to run the code in the correct environment

```shell
jupyter lab code/notebooks/expecon_ms.ipynb &
```

## Publications

In the case you use this code, please cite the following paper: Forster et al., in prep.

## Contributors/Collaborators
* add [`TODO`]
