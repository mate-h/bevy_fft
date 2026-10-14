"""Ablation diagnostics for long-run amplitude growth."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .airy import damp_state_fields, step_airy, wavenumbers
from .decompose import decompose
from .merge import merge_to_state
from .params import Params
from .scenarios import SCENARIOS, params_for
from .step import Sim
from .swe_cmf10 import step_cmf10
from .swe_stelling import cell_q_from_faces, faces_from_cell_q, step_stelling
from .transport import transport


@dataclass(frozen=True)
class Ablation:
    """Flags that neuter hybrid substeps for injector hunting."""

    name: str = "full"
    skip_airy: bool = False
    no_transport_growth: bool = False
    no_div_check: bool = False
    spectral_damp: bool | None = None


ABLATIONS: tuple[Ablation, ...] = (
    Ablation(name="A_full"),
    Ablation(name="B_skip_airy", skip_airy=True),
    Ablation(name="C_no_growth", no_transport_growth=True),
    Ablation(name="D_no_div_check", no_div_check=True),
    Ablation(name="E_damp_off", spectral_damp=False),
)


def midband_energy_frac(field: np.ndarray, dx: float) -> float:
    """Fraction of FFT power in roughly λ ∈ [4dx, 16dx]."""
    n = field.shape[0]
    kx, ky, klen = wavenumbers(n, dx)
    e = np.abs(np.fft.fft2(field)) ** 2
    k_nyq = np.pi / max(dx, 1e-12)
    kn = klen / max(k_nyq, 1e-12)
    # Mid-band: away from DC and away from Nyquist lattice.
    mid = (kn > 0.08) & (kn < 0.35)
    tot = float(e.sum())
    if tot < 1e-30:
        return 0.0
    return float(e[mid].sum() / tot)


def step_ablated(sim: Sim, abl: Ablation) -> Sim:
    """One hybrid step with optional substep neutering."""
    p = sim.p
    (
        sim.h_bar,
        sim.qx_bar,
        sim.qy_bar,
        sim.h_tilde,
        sim.qx_tilde,
        sim.qy_tilde,
    ) = decompose(sim.h, sim.qx, sim.qy, sim.bed, p)

    if p.bulk_solver == "cmf10":
        (
            sim.h_bar,
            sim.qx_bar,
            sim.qy_bar,
            sim.u_face,
            sim.v_face,
        ) = step_cmf10(sim.h_bar, sim.qx_bar, sim.qy_bar, sim.bed, p)
    else:
        sim.u_face, sim.v_face = faces_from_cell_q(
            sim.h_bar, sim.qx_bar, sim.qy_bar
        )
        sim.h_bar, sim.u_face, sim.v_face = step_stelling(
            sim.h_bar, sim.u_face, sim.v_face, sim.bed, p
        )
        sim.qx_bar, sim.qy_bar = cell_q_from_faces(
            sim.h_bar, sim.u_face, sim.v_face
        )

    u_mid = 0.5 * (sim.u_face_prev + sim.u_face)
    v_mid = 0.5 * (sim.v_face_prev + sim.v_face)

    if not abl.skip_airy:
        h_before = sim.h_tilde.copy()
        sim.h_tilde, sim.qx_tilde, sim.qy_tilde = step_airy(
            sim.h_tilde,
            sim.qx_tilde,
            sim.qy_tilde,
            sim.h_bar,
            p,
            h_tilde_prev=sim.h_tilde_prev,
        )
        sim.h_tilde_prev = h_before

    gamma_saved = p.gamma_surf
    if abl.no_transport_growth:
        p.gamma_surf = 0.0
    try:
        sim.h_tilde, sim.qx_tilde, sim.qy_tilde = transport(
            sim.h_tilde,
            sim.qx_tilde,
            sim.qy_tilde,
            sim.h_bar,
            sim.u_face,
            sim.v_face,
            p,
            u_mid=u_mid,
            v_mid=v_mid,
        )
    finally:
        p.gamma_surf = gamma_saved

    sim.h, sim.qx, sim.qy = merge_to_state(
        sim.h,
        sim.h_bar,
        sim.qx_bar,
        sim.qy_bar,
        sim.h_tilde,
        sim.qx_tilde,
        sim.qy_tilde,
        p,
        include_div_check=not abl.no_div_check,
        u_face=sim.u_face,
        v_face=sim.v_face,
    )
    sim.h, sim.qx, sim.qy = damp_state_fields(sim.h, sim.qx, sim.qy, sim.bed, p)
    sim.u_face_prev = sim.u_face.copy()
    sim.v_face_prev = sim.v_face.copy()
    sim.frame += 1
    return sim


def run_ablation(
    abl: Ablation,
    *,
    frames: int = 2400,
    n: int = 128,
    progress: bool = False,
) -> dict:
    scen = next(s for s in SCENARIOS if s.name == "baseline")
    p = params_for(scen, gpu_parity=False)
    p.n = n
    if abl.spectral_damp is not None:
        p.spectral_damp = abl.spectral_damp
    sim = Sim.create(p)
    samples = []
    log_every = max(frames // 20, 1)
    for i in range(frames):
        step_ablated(sim, abl)
        if (i + 1) % log_every == 0 or i == 0 or i + 1 == frames:
            eta = sim.h + sim.bed
            row = {
                "frame": i + 1,
                "sum_h": float(sim.h.sum()),
                "max_abs_eta": float(np.max(np.abs(eta))),
                "max_abs_h_tilde": float(np.max(np.abs(sim.h_tilde))),
                "midband_frac": midband_energy_frac(sim.h_tilde, p.dx),
                "h_max": float(sim.h.max()),
            }
            samples.append(row)
            if progress:
                print(
                    f"[{abl.name}] f={row['frame']} max|η|={row['max_abs_eta']:.3f} "
                    f"sum_h={row['sum_h']:.1f}",
                    flush=True,
                )
    return {
        "ablation": asdict(abl),
        "n": n,
        "frames": frames,
        "params": {
            "mound_amp": p.mound_amp,
            "mound_radius_frac": p.mound_radius_frac,
            "bulk_solver": p.bulk_solver,
            "spectral_damp": p.spectral_damp,
        },
        "final": samples[-1],
        "samples": samples,
        "blew_up": samples[-1]["max_abs_eta"] > 20.0,
    }


def run_diagnose_suite(
    *,
    frames: int = 2400,
    n: int = 128,
    out_dir: Path,
    progress: bool = False,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for abl in ABLATIONS:
        rep = run_ablation(abl, frames=frames, n=n, progress=progress)
        path = out_dir / f"diagnose_{abl.name}.json"
        path.write_text(json.dumps(rep, indent=2) + "\n")
        reports.append(
            {
                "name": abl.name,
                "blew_up": rep["blew_up"],
                "final_max_abs_eta": rep["final"]["max_abs_eta"],
                "final_sum_h": rep["final"]["sum_h"],
                "path": str(path),
            }
        )
        print(
            f"{abl.name}: blew_up={rep['blew_up']} "
            f"max|η|={rep['final']['max_abs_eta']:.3f} "
            f"sum_h={rep['final']['sum_h']:.1f}",
            flush=True,
        )
    summary = {
        "frames": frames,
        "n": n,
        "runs": reports,
        "dominant_hint": _dominant_hint(reports),
    }
    summary_path = out_dir / "diagnose_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _dominant_hint(reports: list[dict]) -> str:
    by = {r["name"]: r for r in reports}
    full = by.get("A_full", {})
    if not full.get("blew_up"):
        return "full pipeline stayed bounded; no injector found at this horizon"
    if by.get("B_skip_airy", {}).get("blew_up") is False:
        return "skipping Airy bounds the run → Airy↔merge feedback primary"
    if by.get("D_no_div_check", {}).get("blew_up") is False:
        return "zeroing merge div_check bounds the run → merge transport flux injector"
    if by.get("C_no_growth", {}).get("blew_up") is False:
        return "disabling transport growth bounds the run → shoaling gain amplifier"
    if by.get("E_damp_off", {}).get("blew_up") and full.get("blew_up"):
        return "damp on/off both blow → mid-band secular mode, not only high-k lattice"
    return "full blew up; ablations inconclusive — inspect per-run samples"
