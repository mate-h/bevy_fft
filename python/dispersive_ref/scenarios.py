"""Named initial conditions for videos and long-horizon gates."""

from __future__ import annotations

from dataclasses import dataclass

from .params import Params


@dataclass(frozen=True)
class Scenario:
    name: str
    mound_amp: float
    mound_radius_frac: float
    description: str
    # NumPy long-horizon amplitude gates: (frame, max|η|).
    long_eta_gates: tuple[tuple[int, float], ...] = ()


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="baseline",
        mound_amp=4.0,
        mound_radius_frac=0.45,
        description="Gentle wide mound (paper-scale rest depth 4 m)",
        long_eta_gates=((1200, 6.0), (2400, 6.0)),
    ),
    Scenario(
        name="steep",
        mound_amp=10.0,
        mound_radius_frac=0.35,
        description="Steeper mound: decompose gradient penalty + Airy stress",
        long_eta_gates=((1200, 16.0),),
    ),
    Scenario(
        name="narrow",
        mound_amp=6.0,
        mound_radius_frac=0.18,
        description="Narrow mound: mid-band dispersive rings",
        long_eta_gates=((1200, 10.0),),
    ),
    Scenario(
        name="ripple",
        mound_amp=3.0,
        mound_radius_frac=0.05,
        description="Compact bump with kh≈1 so Airy gets a real share of the IC",
        long_eta_gates=((1200, 8.0), (2400, 8.0)),
    ),
    Scenario(
        name="deep_drop",
        mound_amp=14.0,
        mound_radius_frac=0.5,
        description="Large free-surface drop (still fully wet)",
        long_eta_gates=((600, 20.0),),
    ),
)


def params_for(scenario: Scenario, gpu_parity: bool = True) -> Params:
    p = Params(
        mound_amp=scenario.mound_amp,
        mound_radius_frac=scenario.mound_radius_frac,
    )
    if gpu_parity:
        p.bulk_solver = "cmf10"
        p.spectral_damp = True
    return p
