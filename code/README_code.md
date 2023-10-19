# ExPeCoN – **code**

`[Last update: October 18, 2023]`

***
    Period:     2023-10 - ...
    Status:     in preparation / work in progress / finalized

    Author(s):  Carina Forster
    Contact:    forster@cbs.mpg.de

***

## Codebase

*Refer to the corresponding code/scripts written for the analysis etc.
Which languages (Python, R, Matlab, ...) were used? Are there specific package versions,
which one should take care of? Or is there a container (e.g., Docker) or virtual environment?*

### Python
Python code is stored in `./code/expecon_ms/` and can be installed as a python package
(see the main [README.md](../README.md)).

R-scripts are stored in `./code/Rscripts/`

#### Jupyter Notebooks
Jupyter notebooks are stored in `./code/notebooks/`.
They are used for data exploration and visualization.

### Configs

Paths to data, parameter settings, etc. are stored in the config file: `./code/configs/config.toml`

In case of a public fork of this project:
Private config files that contain, e.g., passwords, and therefore should not end up in a remote repository
can be listed in: `./code/configs/private_config.toml`, which is ignored by git.

Both files will be read out by the script in `./code/expecon_ms/configs.py`.
Keep both config toml files and the script in their current places.

To use the project configs in Python scripts or notebooks, do the following:

```python
from expecon_ms.configs import config, path_to

# get the path to data
path_to_data = path_to.DATA

# Get parameter from config
weight_decay = config.params.weight_decay

# Get private parameter from config
api_key = config.service_x.api_key
```

### Experiment code
The Matlab-based experiment code is stored in `./code/experimental_code/` and was written by Martin Grund et al.
