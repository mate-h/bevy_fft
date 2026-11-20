"""Steady linear deep-water wave field of a pressure patch moving at U (Havelock / pressure-patch theory).

eta_hat(k) = -(p_hat/rho) * k / (g k - (U kx - i mu)^2)   (boat moving toward -x, waves trail to +x)

Prints the crest-to-trough height H = max - min of eta along lines parallel to the track at lateral
offsets y, optionally limited to a window behind the boat.
"""
import sys
import numpy as np

g, rho = 9.81, 1025.0


def field(U, mass, Lp, Bp, dx=0.25, size=1024.0, mu_frac=0.01, edge=0.5, ny=None):
    n = int(size / dx)
    x = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(x, x, indexing="xy")
    # Smoothed rectangle of length Lp along x, beam Bp along y.
    def soft(u, half):
        return 0.5 * (np.tanh((u + half) / edge) - np.tanh((u - half) / edge))
    shape = soft(X, 0.5 * Lp) * soft(Y, 0.5 * Bp)
    p = shape / (shape.sum() * dx * dx) * mass * g  # N/m^2, integrates to the weight
    kx = 2 * np.pi * np.fft.fftfreq(n, dx)
    KX, KY = np.meshgrid(kx, kx, indexing="xy")
    k = np.hypot(KX, KY)
    k0 = g / U**2
    mu = mu_frac * U * k0
    ph = np.fft.fft2(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = -(1 / rho) * k / (g * k - (U * KX - 1j * mu) ** 2)
    tf[0, 0] = 0.0
    eta = np.real(np.fft.ifft2(ph * tf))
    return x, eta


def heights(x, eta, ys, behind=None):
    out = []
    for y in ys:
        j = np.argmin(abs(x - y))
        row = eta[j]
        if behind is not None:
            sel = (x > -10) & (x < behind)
            row = row[sel]
        out.append(row.max() - row.min())
    return out


if __name__ == "__main__":
    mass = float(sys.argv[1]) if len(sys.argv) > 1 else 4000.0
    ys = [10, 15, 20, 30]
    for kn in (10, 13, 15):
        U = kn * 0.5144
        for Lp, Bp in ((3.5, 2.2), (5.0, 2.4), (7.0, 2.5)):
            x, eta = field(U, mass, Lp, Bp)
            # Sanity check: which side are the waves on?
            j = np.argmin(abs(x - 10))
            front = np.abs(eta[j][x < -20]).max()
            back = np.abs(eta[j][x > 20]).max()
            full = heights(x, eta, ys)
            win = heights(x, eta, ys, behind=58.0)
            print(
                f"{kn:2d} kn U={U:.2f} Fr={U/np.sqrt(g*7.2):.2f} patch {Lp}x{Bp} m: "
                + " ".join(f"y={y}:{h:.3f}" for y, h in zip(ys, full))
                + "  | within 58 m behind: "
                + " ".join(f"{h:.3f}" for h in win)
                + f"  (front {front:.3f} back {back:.3f})"
            )
