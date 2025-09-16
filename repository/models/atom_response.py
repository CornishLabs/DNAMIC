import math
from ndscan.experiment import MHz, A

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