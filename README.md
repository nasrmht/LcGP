# LcGP (Linearly Constrained Gaussian Processes)

LcGP is a Python library for Linearly Constrained and Multi-Output Gaussian Processes. It provides efficient implementations for Multi-Output Gaussian Processes (MOGP) using the Linear Model of Coregionalization (LMC) and Single-Output Gaussian Processes (SoGP).

## Installation

```bash
pip install .
```

## Structure

-   `LcGP.mogp`: Multi-Output Gaussian Processes
    -   `core.py`: Main MOGP regression class
    -   `kernels`: LMC Kernels and Base Kernels
    -   `likelihoods`: Efficient and Naive likelihood computations
-   `LcGP.sogp`: Single-Output Gaussian Processes
    -   `core.py`: Main SoGP regression class
    -   `parallel.py`: Parallel implementation of SoGP

## Usage

### Multi-Output GP

```python
import numpy as np
from LcGP.mogp.core import MOGPR
from LcGP.mogp.kernels.LCMKernel import LMCKernel
from LcGP.mogp.kernels.Kernel import RBFKernel

# Define kernels
rbf = RBFKernel(input_dim=1)
kernel = LMCKernel(base_kernels=[rbf], output_dim=2, rank=[2])

# Initialize model
model = MOGPR(kernel=kernel)

# Fit and Predict
# model.fit(X_train, y_train)
# mean, var = model.predict(X_test)
```

### Single-Output GP

```python
from LcGP.sogp.core import so_GPRegression
from LcGP.sogp.kernels.Kernel import RBFKernel

# Define kernel
rbf = RBFKernel(input_dim=1)

# Initialize model
model = so_GPRegression(kernel=rbf)

# Fit and Predict
# model.fit(X_train, y_train)
# mean, var = model.predict(X_test)
```
