# Wake theory scripts

NumPy checks behind the wake heights the boat and dispersive tests expect. They need Python 3 and NumPy.

- `kelvin.py` computes the steady, linear deep-water wave field of a pressure patch moving at a given speed (Havelock form, with 1% Rayleigh damping), and prints the crest-to-trough height along lines abeam. `tests/boat_handling_gpu.rs` takes its expected heights from it.
- `compare.py` feeds the same model a pressure head read back from the simulation. It expects a folder holding a `meta.txt` and the dumped head.
- `resist.py` estimates the wave resistance the patch implies.
- `barge.py` covers the towed barge in `tests/dispersive_bodies_gpu.rs`.
