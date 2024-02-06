# Expectation Perception Confidence Neural Correlates (`ExPeCoN`)

    Author(s): Carina Forster et al. (2023)
    Status: Code is not yet reviewed
    Last update: Oct 19, 2023
***
## Open ToDos before:
- [ ] add readme/note to folder `code/experiment_code`, referring to author(s)
- [ ] review functionality
- [ ] ...

## Description

Scripts for the manuscript Expectation Perception Confidence Neural Correlates (`ExPeCoN`).

The `ExPeCoN` study investigates stimulus probabilities and the influence on perception and confidence in a
near-threshold somatosensory detection task in two datasets

## Project structure

*A brief description of the folder structure of the project (Where is what?).* [`TODO`]

The project structure is based on [`scilaunch`](https://github.com/SHEscher/scilaunch).

### Data
* data will be uploaded to [OSF](https://osf.io) [`TODO`]

## Install research code as package

First, clone the project to a local folder:

```shell
git clone https://github.com/CarinaFo/ExPeCoN_ms.git
```

Create a conda environment specific to `expecon_ms`:
Make sure you are in the local folder where you cloned the repository into.

```shell
conda env create -f expecon.yml
# the environment is called expecon_3.9
```

And activate it:

```shell
conda activate expecon_3.9

# Add the conda environment to jupyter (notebook)
python -m ipykernel install --user --name=expecon_3.9
```

Install the project code `expecon_ms` as Python package:

```shell
cd expecon_ms
pip install -e .
```

**Note**: The `-e` flag installs the package in editable mode,
i.e., changes to the code will be directly reflected in the installed package.
Moreover, the code keeps its access to the research data in the underlying folder structure.
Thus, the `-e` flag is recommended to use.

*R*-scripts of the analysis are stored in `./code/Rscripts/`.

### Explore data and analysis
Use jupyter notebooks to explore the data and analysis:

### Open code in your prefered editor but make sure to run the code in the environment

```shell
jupyter lab code/notebooks/expecon_ms.ipynb &
```

## Publications

In case you use this code, please cite the following paper: Forster et al., in prep.

## Contributors/Collaborators
