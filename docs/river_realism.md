# Realistic rivers, streams and waterfalls: a research spike

The crate runs rivers on a CMF10-style shallow water grid (Chentanez and Müller 2010) with nested rings, spline borders, Manning friction, a settled cache, a bank checker, waterfall spray, movers, floating bodies, foam and flow detail. This note lists what a real stream or waterfall shows, which of it the crate shows, and the techniques and papers for the rest, then proposes a backlog. River meshing with concurrent binary trees (v1v.27) and the Kurganov–Petrova comparison (v1v.18) are separate spikes. Nothing here is implemented.

## Targets

| Target | Today | Where | Real-time technique |
|---|---|---|---|
| Pools, riffles, step-pools: the surface steps down | Partial | The solver levels water into any bed, but `tools/bake_topobathy.py` carves smooth channels | Bed generator placing steps and pools from slope (Chin and Wohl 2005; Abrahams et al. 1995) |
| Standing waves, hydraulic jumps | Partial | Shocks are captured (`second_order_depth`), bores foam (`src/water/heightfield.wesl`); untested | Test against Bélanger's conjugate depths; undular jumps (Froude below about 1.7) need dispersion |
| Eddies behind rocks | Unverified | MacCormack keeps some vorticity; nothing measures it | Resolve rocks over 2 to 3 cells, test wake length (Jirka 2001), subgrid wakes (Pan et al. 2012) below that |
| Boils below plunges and jumps | Missing | | Procedural divergence sources seeded below high dissipation |
| Foam lines | Covered for convergence | `src/water/foam.rs`, `foam_flow.wesl` (Neyret 2003) | Shear as a source for eddy lines |
| White water in rapids | Partial | Breaking foam on steep bores only | Aeration field from energy loss, driving albedo and opacity (Valle and Pasternack 2006) |
| Waterfall sheet breaking into ligaments | Missing | `src/shallow_water/waterfalls.rs` emits spray and mist only | Ballistic curtain mesh along the lip, breaking up by fall length (Ervine and Falvey 1987) |
| Waterfall mass and momentum | Simplified | The grid still passes the water down the drop (`waterfalls.wesl`), which conserves mass and places the jump within a cell or two; particles are for the eye | CMF10 §2.4.3: reflecting lip face, particles carry the volume |
| Plunge pool aeration and mist | Partial | Mist at the foot; bubbles are bead 2xl | Aeration field plus 2xl |
| Splashes from objects | Covered for bodies | `FloatingBodySettings::splash` | Movers don't splash; reuse the rule |
| Wading ripples at the right speed | Missing | Every wave moves at √(gh) | v1v.16 Airy layer |
| Wet banks that dry | Missing | Left to the ground material | Terrain-space wetness buffer |
| Streams 0.3 to 2 m wide, cobbles, 5 to 30% slope | Missing | Finest grid used is 0.25 m (`examples/pool`) | 0.05 to 0.1 m ring, Ferguson (2007) friction |
| Turbidity and colour | Partial | Fixed Jerlov type (`src/water/surface.rs`) | Advected turbidity scalar |
| Distant river beyond the nest | Partial | Coarse grid and FFT detail | Settled velocity as a flow map (Yu et al. 2009) |

Sound is out of scope.

## Where shallow water stops working

**Free fall.** Shallow water assumes hydrostatic pressure, so a jet in free fall cannot exist in it. The CMF10 answer, already in `.dev/papers/hfFluid`, is to stop the grid at the lip and let particles carry the water. Crimson Desert does the same, per O'Reilly's breakdown in the corpus: "the sim ends at the top of the waterfall". The crate copies the emitter but not the mass transfer. The visible sheet should be a mesh, since particles alone read as spray, not a coherent nappe.

**Steep cascades.** At 30% the bed is 17° off level and cos θ ≈ 0.96, so the hydrostatic error is a few percent. The real problems are drops taller than the flow depth, which the waterfall test catches at 1.5 cells, and friction. In cobble streams h/D84 is near 1. Rickenmann and Recking (2011) find Manning fails below h/D84 of about 10, while Ferguson's (2007) variable-power equation fits the field data across the range.

**Dispersion.** Undular jumps and the ripples that follow a wading leg depend on the speed of short waves, which shallow water gets wrong. Bead v1v.16 adds an Airy layer for ripples. Stationary undular jumps are a steady non-hydrostatic effect. SWASH (Zijlema et al. 2011) adds a non-hydrostatic correction on a staggered grid, at the cost of a Poisson-like solve. Whether the Jeschke and Wojtan split in `src/dispersive/` forms stationary undular jumps is untested and should be checked first.

**Turbulence below grid scale.** Depth-averaged eddies behind a rock are real and, at 2 to 3 cells per rock or more, resolvable. Bottom friction limits them through Jirka's (2001) stability number S = c_f D / h: short wakes in riffles, long ones in pools. That is a cheap test. Below grid scale the options are vorticity confinement or wake synthesis (Pan et al. 2012), turbulence particles (Pfaff et al. 2010) or energy-driven procedural detail (Narain et al. 2008). A scalar dissipation field driving flow detail, foam and boils gives most of the look cheaply. Injecting vorticity into the solver changes the flow, so the eddy test should come first.

**Full 3D.** Chentanez, Müller and Kim (2015) couple a 3D Eulerian patch on restricted tall cells (Chentanez and Müller 2011) to a heightfield and particles. A patch at a plunge pool would give real plunging jets, at the cost of a 3D pressure solve and a second solver. Local SPH or FLIP has the same cost, and no other real-time coupling of it to river shallow water turned up. Treat 3D as a research option, not the plan.

## Game and production references

- **Crimson Desert** (O'Reilly 2026, transcript in the corpus): a live grid near 512² around the player, waterfalls as VFX, objects write only while they move, coarse frozen state when the camera stops. The crate already follows this split.
- **Uncharted 4** (Gonzalez-Ochoa 2016, "Rendering Rapids in Uncharted 4", Advances): rapids from offline simulation baked into layered procedural geometry and wave particles. This is the main reference for the look of white water.
- **Sea of Thieves** (Ang et al. 2018): a time-blurred foam buffer and foam around objects.
- **Far Cry 5** (Cutocheras and Grujic 2018, GDC): sloped rapids and waterfalls with flow maps. GDC Vault only.
- **Unreal Engine** Niagara Fluids and the Water plugin's shallow water river (Epic docs), and Still Wakes the Deep (Epic spotlight): a 2D shallow water grid in Niagara with readback for gameplay. This is the baseline feature set.
- **DreamWorks "Instafalls"** (talk, 2019): procedural waterfalls with foam and ripples triggered where falls land on water.
- No water-specific talk turned up for Ghost of Tsushima, The Last of Us Part II, RDR2 rivers or God of War. Horizon (Malan 2022), Portal 2 (Vlachos 2010) and FluidNinja are already covered by the corpus and the design doc.

## In the corpus but not yet used for rivers

| Paper | Location | Use |
|---|---|---|
| Chentanez and Müller 2010, §2.4.3 | `.dev/papers/hfFluid` | Mass-carrying waterfall particles |
| Chentanez and Müller 2011, restricted tall cells | `.dev/papers/tallCells` | 3D plunge pool spike |
| Chentanez, Müller and Kim 2015 | `.dev/papers/hybridsim_preprinted` | 3D/heightfield/particle coupling at falls |
| Thürey et al. 2007, bubbles and foam in shallow water | `water/thuerey-2007-bubbles-foam-shallow-water` | Plunge pool bubbles (2xl) |
| Wretborn et al. 2022, guided bubbles and wet foam | `water/wretborn-2022-guided-bubbles-wet-foam` | Whitewater classification |
| Burrell et al. 2009, advected river textures | `water/burrell-2009-advected-river-textures` | Regeneration by deformation for river detail |
| Ketefian and Jacobson, cut-cell shallow water boundary | `song-swe-boundary-2011` | Smooth banks instead of stair-steps in narrow streams |
| Canabal et al. 2016, dispersion kernels | `water/canabal-2016-dispersion-kernels` | Airy around rocks (v1v.16 fallback) |
| Dey and Merwade 2016 | `water/dey-merwade-*` | Thalweg on bends |
| Mei 2007, Štava 2008, Jákó 2011, Su 2023 | `/Users/mateh/git/papers` only | Self-shaping beds; low priority |

`/Users/mateh/git/papers` holds no unconverted river or waterfall PDF; its opaque names are clouds, LES, caustics or duplicates.

## Missing papers, ranked

Access: "open" means an author, institutional or arXiv copy was seen in search results. "Paywalled" means the publisher copy is behind a paywall or bot check, for the user to fetch.

| # | Citation | Target | URL | Access |
|---|---|---|---|---|
| 1 | Ferguson 2007, Flow resistance equations for gravel- and boulder-bed streams, WRR 43 W05427 | Friction in cobble streams | https://dro.dur.ac.uk/4448 | Open (Durham) |
| 2 | Chanson 2009, Current knowledge in hydraulic jumps and related phenomena, Eur. J. Mech. B/Fluids 28 | Jump validation, roller aeration | https://core.ac.uk/download/15057462.pdf | Open |
| 3 | Gonzalez-Ochoa 2016, Rendering Rapids in Uncharted 4, SIGGRAPH Advances | White water look | https://advances.realtimerendering.com/s2016/index.html | Open |
| 4 | Yu, Neyret, Bruneton, Holzschuch 2009, Scalable Real-Time Animation of Rivers, CGF 28(2) | Far-field river detail | https://hal.inria.fr/inria-00345903 | Open (HAL) |
| 5 | Ihmsen et al. 2012, Unified Spray, Foam and Air Bubbles for Particle-Based Fluids, Vis. Comput. 28 | Spray/foam/bubble classes at falls | https://cg.informatik.uni-freiburg.de/publications/2012_CGI_sprayFoamBubbles.pdf | Open |
| 6 | Ervine and Falvey 1987, Behaviour of turbulent water jets in the atmosphere and in plunge pools, Proc. ICE 83 | Nappe breakup length, pool penetration | https://doi.org/10.1680/iicep.1987.353 | Paywalled |
| 7 | Valle and Pasternack 2006, Air concentrations of submerged and unsubmerged hydraulic jumps in a bedrock step-pool channel, JGR 111 | Aeration calibration | https://pasternack.ucdavis.edu/research/projects/waterfalls/small-step-hydraulics | Project page open; paper uncertain (Wiley) |
| 8 | Jirka 2001, Large scale flow structures and mixing processes in shallow flows, J. Hydraul. Res. 39(6) | Eddy test behind rocks | https://doi.org/10.1080/00221686.2001.9628285 | Paywalled |
| 9 | Pan, Huang, Tong, Bao 2012, Wake Synthesis for Shallow Water Equation, Pacific Graphics | Subgrid eddies behind small rocks | https://dyingbrain.github.io/assets/wake-2012.pdf | Open; venue from memory |
| 10 | Zijlema, Stelling, Smit 2011, SWASH, Coastal Eng. 58 | Non-hydrostatic correction, undular jumps | https://data-ww3.ifremer.fr/BIB/Zijlema_etal_CE2011.pdf | Open mirror |
| 11 | Rickenmann and Recking 2011, Evaluation of flow resistance in gravel-bed rivers through a large field data set, WRR 47 | Friction validation data | https://doi.org/10.1029/2010WR009793 | Uncertain (Wiley) |
| 12 | Comiti et al. 2007, Field-derived relationships for flow velocity and resistance in high-gradient streams, J. Hydrol. 340 | Step-pool velocity test | https://www.umt.edu/wilcox/documents/publications/comiti-etal2007-jhydrol.pdf | Open |
| 13 | Chin and Wohl 2005, Toward a theory for step pools in stream channels, Prog. Phys. Geog. 29(3) | Step-pool bed generator | https://doi.org/10.1191/0309133305pp449ra | Paywalled |
| 14 | Abrahams, Li, Atkinson 1995, Step-pool streams: adjustment to maximum flow resistance, WRR 31 | Step height/spacing vs slope | https://doi.org/10.1029/95WR01957 | Paywalled; DOI from memory |
| 15 | Holmberg and Wünsche 2004, Modelling of Turbulent Water over Natural Terrain, IVCNZ | Weir model for waterfalls in graphics | https://www-ist.massey.ac.nz/dbailey/sprg/IVCNZ/Proceedings/IVCNZ_55.pdf | Open |
| 16 | Wilcox and Wohl 2006, Flow resistance dynamics in step-pool channels 1, WRR 42 | Spill resistance at step lips | https://www.umt.edu/wilcox/documents/publications/wilcox_wohl-wrr2006.pdf | Open |
| 17 | Chanson and Montes 1995, Characteristics of undular hydraulic jumps, J. Hydraul. Eng. 121(2) | Standing wave wavelength and amplitude | https://core.ac.uk/works/6239037 | Open (UQ) |
| 18 | Pfaff, Thuerey, Cohen, Tariq, Gross 2010, Scalable Fluid Simulation using Anisotropic Turbulence Particles, TOG 29(6) | Subgrid turbulence detail | https://media.disneyanimation.com/uploads/production/publication_asset/27/asset/turbulence-pfaff.pdf | Open |
| 19 | Jeschke et al. 2020, Making Procedural Water Waves Boundary-aware, CGF 39(8) | Detail waves near banks and rocks | https://research-explorer.ista.ac.at/record/8766 | Open |
| 20 | Yu, Neyret, Bruneton, Holzschuch 2011, Lagrangian Texture Advection, TVCG 17(11) | Flow detail without stretching | HAL id to confirm (inria-00355827 is the earlier spectrum-preserving advection report) | Open (HAL) |
| 21 | Mihalef, Metaxas, Sussman 2009, Simulation of Two-Phase Flow with Sub-Scale Droplet and Bubble Effects, CGF 28(2) | When a sheet breaks into drops | https://doi.org/10.1111/j.1467-8659.2009.01362.x | Paywalled |
| 22 | DreamWorks 2019, Instafalls talk | Procedural waterfalls with landing foam | https://research.dreamworks.com/wp-content/uploads/2019/10/talk_dragon3_instafalls_embedded_v5.pdf | Open; authors not checked |

## Proposed backlog

Five epics. IDs are placeholders. Dependencies name existing beads.

### Epic A: stream-scale solver

| Bead | P | Description | Depends on |
|---|---|---|---|
| A1 Run a stream ring at 0.05 to 0.1 m | P2 | A finest ring with at least 6 cells across a 0.5 m stream, substepped to CFL (about 0.025 s at 0.1 m and 3 m/s). Measure a 512² ring on the M5. | v1v.15 |
| A2 Ferguson variable-power friction | P2 | Ferguson's (2007) law with a per-cell D84 texture beside Manning. A GPU test compares reach velocity with the equation. | v1v.14 |
| A3 Step-pool and pool-riffle beds | P2 | Place steps, pools, riffles and boulders from local slope in `tools/bake_topobathy.py` or a bed generator (Abrahams et al. 1995). | v1v.21 |
| A4 Hydraulic jump flume test | P2 | Supercritical inflow into tailwater; conjugate depths within 10% of Bélanger. Also an acceptance scene for v1v.18. | none |
| A5 Shallow wake test behind a boulder | P3 | Recirculation length behind a cylinder as Jirka's (2001) stability number varies. Decides whether subgrid wakes are needed. | A1 |
| A6 Non-hydrostatic spike | P4 | Does the dispersive split form a stationary undular jump? If not, cost a SWASH-style correction. | A4, v1v.16 |

Fold v1v.18 (KP) into this epic, with A4 and A5 as its comparison scenes.

### Epic B: waterfalls

| Bead | P | Description | Depends on |
|---|---|---|---|
| B1 Mass-carrying waterfall particles | P2 | CMF10 §2.4.3: the lip face reflects for the lower cell and particles take its volume and momentum. Spray stays as the visual layer. | v1v.20, hkp |
| B2 Return particle mass to the grid | P2 | Landing mass particles add depth and momentum through the source texture. A test checks inflow equals outflow within 2%. | B1 |
| B3 Waterfall curtain mesh | P3 | A ribbon along each lip on the ballistic path, breaking up after a fall length (Ervine and Falvey 1987). Rework eyk into one ribbon builder for lips and fronts. | B1, eyk |
| B4 Plunge pool aeration | P3 | Impacts feed C1 and bubbles. Rework 2xl to cover plunge-pool bubble plumes. | 2xl, C1 |
| B5 3D plunge patch spike | P4 | Estimate a tall-cell patch (Chentanez, Müller and Kim 2015) under one fall. Paper study only. | B2 |

### Epic C: river surface and rendering

| Bead | P | Description | Depends on |
|---|---|---|---|
| C1 Aeration field | P2 | Air fraction made where flow loses energy, advected and decayed like foam, driving albedo, opacity and spray. Calibrate with Valle and Pasternack (2006). | none |
| C2 Eddy lines and boils | P3 | Shear as a foam source; procedural boils below high dissipation. | C1 |
| C3 Far river detail from the settled cache | P3 | Settled velocity as a flow map outside the nest (Yu et al. 2009). | v1v.19 |
| C4 Wet banks | P3 | Terrain-space wetness from wet cells and landed particles, drying over minutes. | none |
| C5 Turbidity | P4 | Advected turbidity shifting the Jerlov colour. | C1 |

v1v.27 (CBT) and v1v.26 (mesh shaders) stay as the geometry path. C3 and A1 should feed v1v.27's needs.

### Epic D: interaction

| Bead | P | Description | Depends on |
|---|---|---|---|
| D1 Wading splashes | P3 | Movers splash with the floating-body rule. | none |
| D2 v1v.16 Airy on the finest ring | P3 | Keep, scoped to the stream ring. | A1 |
| D3 Debris jams | P4 | Debris piles up on boulders, as in Crimson Desert. | none |

### Epic E: validation scene

| Bead | P | Description | Depends on |
|---|---|---|---|
| E1 `stream` example: mountain stream with a waterfall into a pool | P1 | A 60 m reach, 1 m wide, 15% slope, step-pools, boulders and a 4 m fall into a pool. GPU tests: mass balance within 2%, conjugate depths at a jump, recirculation behind a boulder, foam on the convergence line, mean velocity within 20% of Ferguson. | A1, A2, A3, B1, B2 |
| E2 Stream frame budget | P2 | Per-pass GPU time and a budget for ring, particles and aeration. | E1 |

## Uncertain claims

- The Crimson Desert behaviour is second-hand, from one video breakdown.
- Abrahams' step ratio is from memory: step height over spacing is one to two times the slope. Its DOI also needs checking.
- Ervine and Falvey's breakup lengths are paraphrased from secondary sources.
- It is unverified whether the current advection keeps eddies at 0.1 m cells.
- It is unknown whether the dispersive split forms stationary undular jumps.
- The Pan et al. venue and the Instafalls authors are not confirmed.

## Review

An independent review (2026-11-19) checked this survey against the code and the two spikes, and changed these points.

- None of the 22 missing papers is in `.dev/papers` or `~/git/papers`. The HAL id for entry 20 points to an earlier report, not the TVCG paper.
- Mass-carrying waterfall particles are demoted. Passing water through the grid conserves it. The particle queue holds 4096 slots and drops on overflow (`waterfalls.wesl`), so moving the volume into particles would lose water.
- A standing-jump test against Bélanger would pass today. The real defect is the moving bore, so the bore test leads.
- A fine stream ring at 0.074 m inside ×3 nests needs about 27 substeps a frame, roughly 16 ms at 512². It becomes a decision between a fine ring and a 1D channel along `RiverSpline`.
- The stream example can't be P1 while it waits on P2 work. It is the acceptance scene at P3.

The backlog that came out of this is bead epic bevy_fft-zog. Its parts are solver correctness, the river surface on the mesh, white water and waterfalls, and stream scale.
