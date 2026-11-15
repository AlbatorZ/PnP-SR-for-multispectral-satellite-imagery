import time

import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import ListedColormap

import deepinv as dinv
from deepinv.loss.metric import SSIM

def clean_and_normalize(image):
    """Replace NaN/inf values, then rescale the image to [0, 1] (min-max over the whole image)."""
    image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    return (image - image.min()) / (image.max() - image.min())


def to_rgb(img):
    """(4, H, W) tensor with channels (B, G, R, NIR) -> (H, W, 3) RGB arra
y, gamma-corrected for display."""
    img = clean_and_normalize(img)
    return (img[[2, 1, 0]] ** 0.5).permute(1, 2, 0).numpy()

@torch.no_grad()
def operator_norm_sq(physics, x_shape, device, n_iter=100, tol=1e-6):
    """Estimate ||A||^2 (largest eigenvalue of A^T A) by power iteration.

    ||A||^2 is the Lipschitz constant L of the data-fidelity gradient;
    gradient descent requires the step size GAMMA < 2 / L.
    """
    v = torch.randn(x_shape, device=device)
    v = v / v.norm()
    lam_old = 0.0
    for _ in range(n_iter):
        AtAv = physics.A_adjoint(physics.A(v))
        lam = torch.dot(v.flatten(), AtAv.flatten()).item()
        v = AtAv / AtAv.norm()
        if abs(lam - lam_old) <= tol * abs(lam):
            break
        lam_old = lam
    return lam