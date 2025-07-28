# Circular beam approximation for quantum channels in the turbulent atmosphere

This repository contains the source code used to generate the figures presented in the study [Circular beam approximation for quantum channels in the turbulent atmosphere](https://arxiv.org/abs/2507.12947)

## Example of usage

### Circular Beam Model
```python
import numpy as np
import matplotlib.pyplot as plt
from circular_beam import CircularBeamModel
import analytics

source_w0 = 0.025           # source beam waist [m]
source_wvl = 808e-9         # source wavelength [m]
length = 2500               # atmospherical channel length [m]
rytov2 = 0.25               # Rytov variance (\sigma_R^2)
aperture_radius = 0.01      # receiver aperture radius [m]

# Analytical derivation for beam statistics
source_k = 2 * np.pi / source_wvl
S_BW = np.sqrt(analytics.get_x2_0(source_k, source_w0, length, rytov2))
W2_mean = analytics.get_W2(source_k, source_w0, length, rytov2)
W4_mean = analytics.get_W4(source_k, source_w0, length, rytov2)

# Build model and compute PDT
cb_model = CircularBeamModel.from_beam_params(
    S_BW=S_BW, W2_mean=W2_mean, W4_mean=W4_mean, aperture_radius=aperture_radius
    )
transmittance = np.linspace(0.001, 1, num=100)
cb_pdt = cb_model.get_pdt(transmittance)

plt.plot(transmittance, cb_pdt)
plt.savefig("cb_example.png")
```

### Circular Beam model with transmittance matching method
In addition to the previous code:

```python
from circular_beam import AnchoredCircularBeamModel

# Analytical expressions for trasmittance moments
eta_mean = analytics.get_etha(source_k, source_w0, length, rytov2, aperture_radius)
eta2_mean = analytics.get_etha2(source_k, source_w0, length, rytov2, aperture_radius)

# Build model and compute PDT
cbm_model = AnchoredCircularBeamModel.from_beam_params(
    S_BW=S_BW, eta_mean=eta_mean, eta2_mean=eta2_mean, aperture_radius=aperture_radius,
    initial_guess_W2_mean=W2_mean, initial_guess_W4_mean=W4_mean
    )
transmittance = np.linspace(0.001, 1, num=100)
cbm_pdt = cbm_model.get_pdt(transmittance)

plt.plot(transmittance, cbm_pdt)
plt.savefig("cbm_example.png")
```

## Repository structure

The project is organised as follows:
- `Figure_X.py` - scripts for generating the corresponding figures, where X denotes the figure number in the publication;
- `circular_beam.py`, `beam_wandering.py`, `analytics.py` - auxiliary modules used by the figure scripts;
- `requirements.txt` - a list of required Python packages.


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




