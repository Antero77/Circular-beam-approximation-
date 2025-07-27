# Circular beam approximation for quantum channels in the turbulent atmosphere

Source code for the generation of figures in [Circular beam approximation for quantum channels in the turbulent atmosphere](https://arxiv.org/abs/2507.12947)

## Structure

The project consists of the following items:
- `Figure_X.py` - these files allow one to generate the corresponding figures with number X;
- `circular_beam.py`, `beam_wandering.py`, `analytics.py` - these are auxiliary files, which are used by `Figure_X.py`;
- `requirements.txt` - provides required packages;


## Installation
Make sure you have [Python 3.8+](https://realpython.com/installing-python/#how-to-install-python-on-linux) installed.
Create a [virtual environment](https://packaging.python.org/en/latest/tutorials/installing-packages/#creating-and-using-virtual-environments) and [install the required packages](https://packaging.python.org/en/latest/tutorials/installing-packages/#use-pip-for-installing) from the `requirements.txt` file:
```
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

Since simulation can be time-consuming, we strongly recommend using a GPU for simulation. Thus, you need to [install](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation) `CUDA` on your machine.
Additionally, the `cupy` Python package is required. For example, for CUDA 11.0:
```bash
pip install cupy-cuda110
```
Appropriate lines enabling the use of GPU simulation are already written where they are needed:
```python
from pyatmosphere import gpu
gpu.config['use_gpu'] = True
```





