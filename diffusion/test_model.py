"""Checks of the torch face padding and of the network shapes.

    python diffusion/test_model.py
"""

import numpy as np
import torch

import healpix_faces as hf
from model import Denoiser, Padding

if __name__ == "__main__":
    n = 16
    m = np.random.default_rng(0).normal(size=(2, 12 * n * n))
    pad = Padding()
    pad.mode = "healpix"
    y = pad(torch.tensor(hf.to_faces(m)).reshape(24, 1, n, n)).numpy().reshape(2, 12, n + 2, n + 2)
    assert np.array_equal(y, m[:, hf.pad_nest(n, 1)])
    print("torch healpix padding equals pad_nest")
    d = Denoiser(channels=(32, 64, 64))
    for mode in ("zeros", "healpix"):
        d.net.set_padding(mode)
        out = d(torch.randn(24, 3, n, n), torch.ones(24), torch.randn(24, 1, n, n))
        assert out.shape == (24, 3, n, n)
    print("denoiser shapes ok in both padding modes")
