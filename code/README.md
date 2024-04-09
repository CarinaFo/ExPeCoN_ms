# ExPeCoN – **code**

***
    Author(s): Carina Forster et al.
    Contact: forster@cbs.mpg.de
    Status: work in progress
    Last update: March 26, 2024

***

## Codebase

### Python
Python code is stored in `./code/expecon_ms/` and can be installed as a python package
(see the main [README.md](../README.md)).

R-scripts are stored in `./code/Rscripts/`

#### Jupyter Notebooks
Jupyter notebooks are stored in `./code/notebooks/`.
They are used for data exploration and visualization.

We use nbstripout for seamless [version control](https://towardsdatascience.com/enhancing-data-science-workflows-mastering-version-control-for-jupyter-notebooks-b03c839e25ec) of jupyter notebooks:

nbstripout integrates with Git hooks to automatically strip output cells fromnotebooks when they are committed.
It modifies the notebook’s JSON content, removing the output fields, thus reducing the file size and simplifying diffs.

### Configs

Paths to data, parameter settings, etc. are stored in the config file: `./code/configs/config.toml`

In case of a public fork of this project:
Private config files that contain, e.g., passwords, and therefore should not end up in a remote repository
can be listed in: `./code/configs/private_config.toml`, which is ignored by git.

Both files will be read out by the script in `./code/expecon_ms/configs.py`.
Keep both config toml files and the script in their current places.

To use the project configs in Python scripts or notebooks, do the following:

```python
from expecon_ms.configs import config, paths, params

# get the path to data
path_to_data = paths.DATA

# Get parameter from config
weight_decay = params.weight_decay

# Get private parameter from config
api_key = config.service_x.api_key
```

### Experiment code
The Matlab-based experiment code is stored in `./code/experimental_code/` and was written by Martin Grund et al.
