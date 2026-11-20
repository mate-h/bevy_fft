"""The towed barge test: steady linear field of a 1 m head over its 4 m (along track) by 8 m
footprint at 4 m/s, against the sim's patch, max |eta| more than 10 m from it, behind it."""
import sys
import numpy as np

g = 9.81
d = sys.argv[1]
n = 128
eta = np.fromfile(f"{d}/eta.f32", dtype="<f4").reshape(n, n)
U = 4.0
M, dx = 2048, 0.5
xs = (np.arange(M) - M // 2) * dx
X, Y = np.meshgrid(xs, xs, indexing="xy")
for head_m in (1.0,):
    H = ((abs(X) <= 2.0) & (abs(Y) <= 4.0)).astype(float) * head_m
    kx = 2 * np.pi * np.fft.fftfreq(M, dx)
    KX, KY = np.meshgrid(kx, kx, indexing="xy")
    k = np.hypot(KX, KY)
    # Waves trail toward -x: the barge moves along +x.
    mu = 0.01 * g / U
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = -g * k / (g * k - (U * KX + 1j * mu) ** 2)
    tf[0, 0] = 0
    # Band-limit like the patch's 1 m cells: zero past kn 0.5 and a soft roll-off from 0.28.
    kn = k / np.pi
    t = np.clip((kn - 0.28) / 0.22, 0, 1)
    damp = np.where(kn > 0.5, 0.0, np.exp(-8 * t * t))
    th = np.real(np.fft.ifft2(np.fft.fft2(H) * tf * damp))
    region = (np.hypot(X, Y) > 10) & (X < 0) & (X > -40)
    print(f"theory head {head_m} m: max |eta| {np.abs(th[region]).max():.2f} m, crest to trough {np.ptp(th[region]):.2f} m")
c = (np.arange(n) - n / 2 + 0.5)
CX, CY = np.meshgrid(c, c, indexing="xy")
region = (np.hypot(CX, CY) > 10) & (CX < 0) & (CX > -40)
print(f"sim: max |eta| {np.abs(eta[region]).max():.2f} m, crest to trough {np.ptp(eta[region]):.2f} m")
