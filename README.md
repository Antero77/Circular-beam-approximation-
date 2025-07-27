# Circular beam approximation for quantum channels in the turbulent atmosphere

This repository contains the source code used to generate the figures presented in the study [Circular beam approximation for quantum channels in the turbulent atmosphere](https://arxiv.org/abs/2507.12947)

## Repository structure

The project is organised as follows:
- `Figure_X.py` - these files allow one to generate the corresponding Figures, where X denotes the number of the figure;
- `circular_beam.py`, `beam_wandering.py`, `analytics.py` - these are auxiliary files, which are used by `Figure_X.py`;
- `requirements.txt` - provides required packages.


## Installation instructions
Ensure that [Python 3.8 or later](https://realpython.com/installing-python/#how-to-install-python-on-linux) is installed.
It is recommended to create a [virtual environment](https://packaging.python.org/en/latest/tutorials/installing-packages/#creating-and-using-virtual-environments) and [install the required packages](https://packaging.python.org/en/latest/tutorials/installing-packages/#use-pip-for-installing) from the `requirements.txt` file as follows:
```
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

Due to the computational intensity of the simulations, GPU acceleration is strongly recommended. To enable this, [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation) must be installed on the system, along with the appropriate `cupy` package. For example, for CUDA 11.0: 
```bash
pip install cupy-cuda110
```
GPU-based computation is already supported in the codebase. Relevant lines such as:
```python
from pyatmosphere import gpu
gpu.config['use_gpu'] = True
```

## Usage
To generate a specific figure (*Figure X*), execute the corresponding script `Figure_X.py`.

Scripts such as `Figure_4a-b.py` and `Figure_5a-b.py` produce only a single subfigure (i.e., Figure 4a or 4b, respectively). To generate each subfigure, run the script with the appropriate parameters as specified in the [associated article](https://arxiv.org/abs/2507.12947). For all other figures, the scripts include all necessary parameters by default.

 


