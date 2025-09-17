import math
from ndscan.experiment import MHz, A
import numpy as np
import matplotlib.pyplot as plt

__all__ = ["p_bright_detuned_rabi","image_from_probs_and_locs"]  # optional, limits what gets imported by *

def p_bright_detuned_rabi(freq_Hz: float, coil_current_A: float, rabi_Hz: float, duration_s: float) -> float:
    """SI in, probability out. p = (Ω/Ω_eff)^2 * sin^2(Ω_eff t / 2) with Ω=2π rabi, Δ=2π(f-f0)."""
    f0_Hz = 10*MHz + (0.13*(MHz/A))*coil_current_A
    Omega = 2 * math.pi * max(0.0, rabi_Hz)
    Delta = 2 * math.pi * (freq_Hz - f0_Hz)
    Omega_eff = math.hypot(Omega, Delta)
    if Omega_eff == 0.0:
        return 0.0
    p = (Omega / Omega_eff) ** 2 * (math.sin(0.5 * Omega_eff * max(0.0, duration_s)) ** 2)
    return 0.0 if p < 0.0 else (1.0 if p > 1.0 else p)


def _gaussian2d(shape, x0, y0, sigma):
    yy, xx = np.indices(shape)
    g = np.exp(-((xx - x0)**2 + (yy - y0)**2) / (2.0 * sigma**2))
    s = g.sum()
    if s > 0:
        g /= s
    return g

def image_from_probs_and_locs(ls, shape=(64, 64), muB=1500, muD=200, sigma=1.4, seed=None):
    rng = np.random.default_rng(seed)
    image = rng.poisson(muD, size=shape).astype(np.int32)
    for (x, y, p_bright) in ls:
        if rng.random() < p_bright:
            amp = rng.poisson(muB)
            if amp > 0:
                psf = _gaussian2d(shape, x, y, sigma)
                expected = amp * psf
                image += rng.poisson(expected).astype(np.int32)
    return image

def demo():
    """Optional helper so you can run `python module.py` or call demo() from a notebook."""
    ls = [(16.0, 16.0, 0.95), (32.0, 40.0, 0.9999), (48.0, 20.0, 1.0)]
    img = image_from_probs_and_locs(ls, shape=(64, 64), muB=1500, muD=200, sigma=1.4, seed=42)
    plt.figure(figsize=(5, 5))
    plt.imshow(img, origin='lower', interpolation='nearest')
    plt.title('Simulated atoms in optical tweezers')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.colorbar(label='photons')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demo()
