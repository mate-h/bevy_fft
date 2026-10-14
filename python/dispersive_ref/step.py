"""One hybrid time step = Algorithm 1 (paper)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .airy import damp_state_fields, step_airy
from .decompose import decompose
from .init_basin import init_basin
from .merge import merge_to_state
from .params import Params
from .swe_cmf10 import step_cmf10
from .swe_stelling import cell_q_from_faces, faces_from_cell_q, step_stelling
from .transport import transport


@dataclass
class Sim:
    p: Params
    h: np.ndarray
    qx: np.ndarray
    qy: np.ndarray
    bed: np.ndarray
    h_bar: np.ndarray
    qx_bar: np.ndarray
    qy_bar: np.ndarray
    h_tilde: np.ndarray
    qx_tilde: np.ndarray
    qy_tilde: np.ndarray
    u_face: np.ndarray
    v_face: np.ndarray
    # Previous surface height for Alg. 2 time un-stagger average.
    h_tilde_prev: np.ndarray
    # Previous bulk face velocity for Alg. 3 midpoint ū.
    u_face_prev: np.ndarray
    v_face_prev: np.ndarray
    frame: int = 0

    @property
    def n(self) -> int:
        return self.p.n

    @classmethod
    def create(cls, p: Params | None = None) -> "Sim":
        p = p or Params()
        h, qx, qy, bed = init_basin(p)
        z = np.zeros((p.n, p.n), dtype=np.float64)
        return cls(
            p=p,
            h=h,
            qx=qx,
            qy=qy,
            bed=bed,
            h_bar=h.copy(),
            qx_bar=z.copy(),
            qy_bar=z.copy(),
            h_tilde=z.copy(),
            qx_tilde=z.copy(),
            qy_tilde=z.copy(),
            u_face=z.copy(),
            v_face=z.copy(),
            h_tilde_prev=z.copy(),
            u_face_prev=z.copy(),
            v_face_prev=z.copy(),
        )


def step(sim: Sim) -> Sim:
    """Advance one frame in place and return sim."""
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

    h_before_airy = sim.h_tilde.copy()
    sim.h_tilde, sim.qx_tilde, sim.qy_tilde = step_airy(
        sim.h_tilde,
        sim.qx_tilde,
        sim.qy_tilde,
        sim.h_bar,
        p,
        h_tilde_prev=sim.h_tilde_prev,
    )
    # Alg. 2 companion: previous half-step h̃ for the next average.
    sim.h_tilde_prev = h_before_airy

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

    sim.h, sim.qx, sim.qy = merge_to_state(
        sim.h,
        sim.h_bar,
        sim.qx_bar,
        sim.qy_bar,
        sim.h_tilde,
        sim.qx_tilde,
        sim.qy_tilde,
        p,
        u_face=sim.u_face,
        v_face=sim.v_face,
    )
    # Clear static high-k leftovers that survive in total h when q̃ is zeroed
    # each decompose (Airy damp alone never rewrites the merged height).
    sim.h, sim.qx, sim.qy = damp_state_fields(sim.h, sim.qx, sim.qy, sim.bed, p)

    sim.u_face_prev = sim.u_face.copy()
    sim.v_face_prev = sim.v_face.copy()
    sim.frame += 1
    return sim
