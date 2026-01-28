import pytest
import numpy as np
from LcGP.mogp.core import MOGPR
from LcGP.sogp.core import so_GPRegression
from LcGP.mogp.kernels.LCMKernel import LMCKernel
from LcGP.mogp.kernels.Kernel import RBFKernel as MoGP_RBFKernel
from LcGP.sogp.kernels.Kernel import RBFKernel as SoGP_RBFKernel

def test_imports():
    assert MOGPR is not None
    assert so_GPRegression is not None
    assert LMCKernel is not None
    assert MoGP_RBFKernel is not None
    assert SoGP_RBFKernel is not None

def test_sogp_init():
    kernel = SoGP_RBFKernel(length_scale=1.0)
    model = so_GPRegression(kernel=kernel)
    assert model is not None

def test_mogp_init():
    rbf = MoGP_RBFKernel(input_dim=1)
    kernel = LMCKernel(base_kernels=[rbf], output_dim=2, rank=[2])
    model = MOGPR(kernel=kernel)
    assert model is not None
