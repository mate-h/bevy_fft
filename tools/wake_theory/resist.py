"""Havelock wave resistance of the sim's own pressure head, R = rho g ∫ head d(eta)/dx' dA (x' aft),
at several speeds, scaled to the boat's weight."""
import sys
import numpy as np

src = open(sys.argv[0].replace("resist.py", "compare.py")).read().split("kx = 2 * np.pi")[0]
exec(src)
rho = 1025.0
mass = 4492.0
H = H * (mass / rho) / (H.sum() * dx * dx)
kx = 2 * np.pi * np.fft.fftfreq(M, dx)
KX, KY = np.meshgrid(kx, kx, indexing="xy")
k = np.hypot(KX, KY)
Hh = np.fft.fft2(H)
for U in (3.0, 4.0, 4.6, 5.5, 6.5, 7.7, 9.0):
    mu = 0.01 * g / U
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = -g * k / (g * k - (U * KX - 1j * mu) ** 2)
    tf[0, 0] = 0
    eta_t = np.real(np.fft.ifft2(Hh * tf))
    deta = np.gradient(eta_t, dx, axis=1)
    R = rho * g * np.sum(H * deta) * dx * dx
    print(f"U {U:.1f} m/s Fr {U/np.sqrt(g*7.2):.2f}: wave resistance {R/1000:.2f} kN, R/W {R/(mass*g):.3f}")
