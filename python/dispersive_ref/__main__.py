"""CLI for the NumPy dispersive reference.

Examples:
  python -m dispersive_ref video --scenario ripple --frames 2400 --stride 10 \\
    --out ../target/dispersive_ref/height_ripple_10x.mp4
  python -m dispersive_ref validate-long --scenario baseline --n 256
  python -m dispersive_ref validate-dispersion
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python -m dispersive_ref` from repo root or python/.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dispersive_ref.diagnose import run_diagnose_suite
from dispersive_ref.dispersion import dispersion_suite
from dispersive_ref.io_dump import write_sim_dump
from dispersive_ref.params import Params
from dispersive_ref.scenarios import SCENARIOS, params_for
from dispersive_ref.step import Sim, step
from dispersive_ref.visualize import run_height_history, write_height_video


def _params_from_args(args: argparse.Namespace) -> Params:
    p = Params(
        n=getattr(args, "n", 256),
        tile_world=getattr(args, "tile_world", 256.0),
        dt=getattr(args, "dt", 1.0 / 60.0),
        diffusion_iters=getattr(args, "diffusion_iters", 128),
        mound_amp=getattr(args, "mound_amp", 4.0),
        mound_radius_frac=getattr(args, "mound_radius", 0.45),
    )
    if getattr(args, "gpu_parity", False):
        p.spectral_damp = True
        p.bulk_solver = "cmf10"
    if getattr(args, "bulk", None):
        p.bulk_solver = args.bulk
    return p


def cmd_run(args: argparse.Namespace) -> int:
    p = _params_from_args(args)
    sim = Sim.create(p)
    for i in range(args.frames):
        step(sim)
        if args.progress and (i + 1) % max(args.frames // 10, 1) == 0:
            print(f"frame {i+1}/{args.frames} h_max={sim.h.max():.4f}", flush=True)
    out = Path(args.out)
    write_sim_dump(out, sim)
    print(f"wrote {out} (frames={args.frames}, h_max={sim.h.max():.6f})")
    return 0


def cmd_video(args: argparse.Namespace) -> int:
    if args.scenario:
        scen = next(s for s in SCENARIOS if s.name == args.scenario)
        p = params_for(scen, gpu_parity=getattr(args, "gpu_parity", False))
        p.n = getattr(args, "n", p.n)
        p.tile_world = getattr(args, "tile_world", p.tile_world)
        p.dt = getattr(args, "dt", p.dt)
        p.diffusion_iters = getattr(args, "diffusion_iters", p.diffusion_iters)
        if getattr(args, "bulk", None):
            p.bulk_solver = args.bulk
        title = f"dispersive_ref · {scen.name}"
    else:
        p = _params_from_args(args)
        title = "dispersive_ref · mound collapse"
    history, times = run_height_history(p, args.frames, progress=args.progress)
    out = write_height_video(
        Path(args.out),
        history,
        times,
        fps=args.fps,
        stride=args.stride,
        cmap=args.cmap,
        title=title,
    )
    print(
        f"wrote {out} (sim_frames={args.frames}, video_frames={len(history[::max(args.stride,1)])}, "
        f"η∈[{history.min():.3f},{history.max():.3f}])"
    )
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    summary = run_diagnose_suite(
        frames=args.frames,
        n=args.n,
        out_dir=Path(args.out),
        progress=args.progress,
    )
    print(json.dumps(summary, indent=2))
    print("hint:", summary["dominant_hint"])
    return 0


def cmd_validate_dispersion(args: argparse.Namespace) -> int:
    report = dispersion_suite(Params())
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("PASS" if report["passed"] else "FAIL", "dispersion")
    return 0 if report["passed"] else 1


def cmd_validate_long(args: argparse.Namespace) -> int:
    """Long-horizon amplitude and volume gates."""
    import numpy as np

    scen = next(s for s in SCENARIOS if s.name == args.scenario)
    if not scen.long_eta_gates:
        print(f"scenario {scen.name} has no long_eta_gates")
        return 1
    p = params_for(scen, gpu_parity=args.gpu_parity)
    p.n = args.n
    if args.bulk:
        p.bulk_solver = args.bulk
    sim = Sim.create(p)
    h0 = float(sim.h.sum())
    max_frame = max(fr for fr, _ in scen.long_eta_gates)
    samples = []
    failed = []
    gates = dict(scen.long_eta_gates)
    for i in range(max_frame):
        step(sim)
        fr = i + 1
        if fr in gates or (args.progress and fr % max(max_frame // 10, 1) == 0):
            eta = sim.h + sim.bed
            row = {
                "frame": fr,
                "max_abs_eta": float(np.max(np.abs(eta))),
                "sum_h": float(sim.h.sum()),
                "volume_drift": float(sim.h.sum() - h0),
            }
            if fr in gates:
                gate = gates[fr]
                row["gate"] = gate
                row["passed"] = row["max_abs_eta"] <= gate and abs(row["volume_drift"]) <= 1.0
                if not row["passed"]:
                    failed.append(fr)
                samples.append(row)
                print(
                    f"f={fr} max|η|={row['max_abs_eta']:.4f} gate={gate} "
                    f"dV={row['volume_drift']:.3e} "
                    f"{'PASS' if row['passed'] else 'FAIL'}",
                    flush=True,
                )
    out = {
        "scenario": scen.name,
        "n": p.n,
        "bulk_solver": p.bulk_solver,
        "samples": samples,
        "passed": not failed,
        "failed_frames": failed,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(out, indent=2) + "\n")
    print("PASS" if out["passed"] else f"FAIL frames={failed}", "→", report_path)
    return 0 if out["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="NumPy reference for dispersive hybrid waves")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--n", type=int, default=256)
        sp.add_argument("--tile-world", type=float, default=256.0)
        sp.add_argument("--dt", type=float, default=1.0 / 60.0)
        sp.add_argument("--diffusion-iters", type=int, default=128)
        sp.add_argument("--mound-amp", type=float, default=4.0)
        sp.add_argument("--mound-radius", type=float, default=0.45)
        sp.add_argument(
            "--gpu-parity",
            action="store_true",
            help="CMF10 bulk + spectral_damp (closest to Bevy GPU)",
        )
        sp.add_argument(
            "--bulk",
            choices=("stelling", "cmf10"),
            default=None,
            help="Override bulk SWE solver",
        )

    run_p = sub.add_parser("run", help="Run reference and write a checkpoint dump")
    add_common(run_p)
    run_p.add_argument("--frames", type=int, default=240)
    run_p.add_argument("--out", type=str, default="target/dispersive_ref/latest")
    run_p.add_argument("--progress", action="store_true")
    run_p.set_defaults(func=cmd_run)

    vid_p = sub.add_parser(
        "video", help="Run an example sim and write η(x,y,t) as MP4 or GIF"
    )
    add_common(vid_p)
    vid_p.add_argument("--frames", type=int, default=240)
    vid_p.add_argument(
        "--out",
        type=str,
        default="../target/dispersive_ref/height.mp4",
    )
    vid_p.add_argument("--fps", type=float, default=30.0)
    vid_p.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Keep every Nth sim frame in the video (default 2)",
    )
    vid_p.add_argument("--cmap", type=str, default="viridis")
    vid_p.add_argument(
        "--scenario",
        choices=[s.name for s in SCENARIOS],
        default=None,
        help="Named IC from scenarios.py (overrides mound amp/radius)",
    )
    vid_p.add_argument("--progress", action="store_true")
    vid_p.set_defaults(func=cmd_video)

    diag_p = sub.add_parser(
        "diagnose",
        help="Ablate hybrid substeps and log volume / max|η| over a long run",
    )
    diag_p.add_argument("--frames", type=int, default=2400)
    diag_p.add_argument("--n", type=int, default=128)
    diag_p.add_argument(
        "--out", type=str, default="../target/dispersive_ref/diagnose"
    )
    diag_p.add_argument("--progress", action="store_true")
    diag_p.set_defaults(func=cmd_diagnose)

    disp_p = sub.add_parser(
        "validate-dispersion", help="Paper Eq.2+27 β correction self-check"
    )
    disp_p.add_argument(
        "--report", type=str, default="target/dispersive_validate/dispersion.json"
    )
    disp_p.set_defaults(func=cmd_validate_dispersion)

    long_p = sub.add_parser(
        "validate-long",
        help="Long-horizon max|η| / volume gates (catches Airy blow-up)",
    )
    long_p.add_argument(
        "--scenario",
        choices=[s.name for s in SCENARIOS],
        default="baseline",
    )
    long_p.add_argument("--n", type=int, default=128)
    long_p.add_argument("--gpu-parity", action="store_true")
    long_p.add_argument(
        "--bulk",
        choices=("stelling", "cmf10"),
        default=None,
    )
    long_p.add_argument(
        "--report",
        type=str,
        default="../target/dispersive_validate/long_horizon.json",
    )
    long_p.add_argument("--progress", action="store_true")
    long_p.set_defaults(func=cmd_validate_long)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
