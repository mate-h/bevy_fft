"""Steady linear deep-water response to the sim's own pressure head, against the sim's patch."""
import sys
import numpy as np

g = 9.81
d = sys.argv[1]
n, cell, cx, cz, bx, bz, fx, fz, U = [float(v) for v in open(f"{d}/meta.txt").read().split()]
n = int(n)
eta = np.fromfile(f"{d}/eta.f32", dtype="<f4").reshape(n, n)  # [row j along z][i along x]
head = np.fromfile(f"{d}/head.f32", dtype="<f4").reshape(n, n)
print(f"U {U:.2f} head volume {head.sum()*cell*cell:.2f} m3, peak {head.max():.2f}")

# Rotate into the boat frame: boat heads along -x' (forward), so put the head on a big grid with
# x' = -(along forward), y' = lateral. Sample head by bilinear interpolation.
M = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
dx = cell
xs = (np.arange(M) - M // 2) * dx
XP, YP = np.meshgrid(xs, xs, indexing="xy")
f = np.array([fx, fz]); s = np.array([-fz, fx])


def world(xp, yp):
    # x' positive is behind the boat
    return bx - f[0] * xp + s[0] * yp, bz - f[1] * xp + s[1] * yp


def bilinear(v, wx, wz):
    gx = (wx - cx) / cell + 0.5 * n - 0.5
    gz = (wz - cz) / cell + 0.5 * n - 0.5
    i0 = np.floor(gx).astype(int); j0 = np.floor(gz).astype(int)
    tx = gx - i0; tz = gz - j0
    out = np.zeros_like(gx)
    inside = (i0 >= 0) & (j0 >= 0) & (i0 < n - 1) & (j0 < n - 1)
    i0c = np.clip(i0, 0, n - 2); j0c = np.clip(j0, 0, n - 2)
    a = v[j0c, i0c] * (1 - tx) + v[j0c, i0c + 1] * tx
    b = v[j0c + 1, i0c] * (1 - tx) + v[j0c + 1, i0c + 1] * tx
    out = a * (1 - tz) + b * tz
    return np.where(inside, out, 0.0)


near = (abs(XP) < 15) & (abs(YP) < 15)
H = np.zeros((M, M))
wx, wz = world(XP[near], YP[near])
H[near] = bilinear(head, wx, wz)
print(f"resampled head volume {H.sum()*dx*dx:.2f}")
kx = 2 * np.pi * np.fft.fftfreq(M, dx)
KX, KY = np.meshgrid(kx, kx, indexing="xy")
k = np.hypot(KX, KY)
mu = 0.01 * U * g / U**2
with np.errstate(divide="ignore", invalid="ignore"):
    tf = -g * k / (g * k - (U * KX - 1j * mu) ** 2)
tf[0, 0] = 0
theory = np.real(np.fft.ifft2(np.fft.fft2(H) * tf))

margin = 16 * cell
half = 0.5 * n * cell - margin
for lat in (5, 7.5, 10, 12.5, 15, 20):
    res = []
    for sign in (1, -1):
        back = np.arange(-5, 200, 0.25)
        wx, wz = world(back, sign * lat * np.ones_like(back))
        ok = (abs(wx - cx) <= half) & (abs(wz - cz) <= half)
        stop = np.argmin(ok) if not ok.all() else len(ok)
        b = back[:stop]
        sim = bilinear(eta, wx[:stop], wz[:stop])
        j = np.argmin(abs(xs - sign * lat))
        th_row = theory[j]
        sel = (xs >= -5) & (xs <= b[-1])
        th_win = th_row[sel]
        th_full = th_row[xs >= -5]
        res.append((np.ptp(sim), np.ptp(th_win), np.ptp(th_full), b[-1]))
    r = np.mean(res, axis=0)
    print(f"{lat:5.1f} m abeam: sim {r[0]:.3f}  theory(same window) {r[1]:.3f}  theory(full) {r[2]:.3f}  ratio {r[1]/r[0]:.2f}  window {r[3]:.0f} m")
np.save(f"{d}/theory.npy", theory)
