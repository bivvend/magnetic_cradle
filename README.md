# magnetic_cradle
Magnetic newtons cradle
# Magnetic Newton’s Cradle (Simulation)

A Python simulation of a **Newton’s cradle–like system where the collisions are replaced by magnetic repulsion**.  
Each bob is modelled as a **planar pendulum** with a **rigidly mounted magnetic dipole**, interacting via a softened **3D dipole–dipole force**.

The project is designed to be **modular and extendable**, so different interaction models (dipoles, empirical force laws, nearest-neighbour coupling, damping terms, etc.) can be swapped in cleanly.

---

## Motivation

A traditional Newton’s cradle relies on:
- rigid-body contact,
- short-duration impulses,
- near-elastic collisions.

Replacing contact with **magnetic repulsion** raises interesting questions:
- Can momentum/energy still transfer “end to end”?
- How does interaction range affect impulse-like behaviour?
- When does the system behave collectively rather than discretely?
- How close can a continuous repulsive force come to a collision model?

This project explores those questions using a **minimal but physically grounded model**.

---

## Physical Model

### Pendulum dynamics
- Each bob has **one degree of freedom**: angle `θᵢ(t)` in a vertical plane.
- Motion is constrained to the plane (stiff wires / rods).
- Gravity and optional pivot damping are included.

Bob position:
\[
\mathbf r_i =
\begin{bmatrix}
i d + L\sin\theta_i \\
- L\cos\theta_i \\
0
\end{bmatrix}
\]

### Magnetic model
- Each magnet is treated as a **point magnetic dipole**.
- Dipoles are **rigidly mounted** (no independent spin).
- Dipole direction is **tangential to the motion**.
- Dipole signs alternate (`+ − + − …`) to represent alternating magnetisation.

Dipole vector:
\[
\mathbf m_i(\theta_i) = s_i\, m_d
\begin{bmatrix}
\cos\theta_i \\
\sin\theta_i \\
0
\end{bmatrix}
\]

### Dipole–dipole interaction
- Full 3D dipole–dipole force law is used.
- A **softening length ε** regularises the near-field to avoid singularities.
- Interactions can be:
  - all-to-all (physical),
  - nearest-neighbour only (more “cradle-like”).

---

## Project Structure

.
├── magnetic_cradle.py # Core simulation (physics + integrator)
├── visualise_magnetic_cradle.py # Plotting + animation (mp4 export)
├── README.md
└── examples/


---

## Key Features

- ✅ Modular, class-based architecture
- ✅ Extendable interaction system
- ✅ Stable RK4 time integration
- ✅ Alternating dipole magnetisation
- ✅ Energy-diagnostic–friendly
- ✅ Matplotlib animation with **MP4 export**
- ✅ Designed for experimentation and parameter sweeps

---

## Installation

### Requirements
- Python 3.9+
- NumPy
- Matplotlib
- FFmpeg (for MP4 export)

Install Python dependencies:
```bash
pip install numpy matplotlib


Running a Simulation

In magnetic_cradle.py:

chain = PlanarPendulumChain(
    params=params,
    dipole_magnitude=0.15,
    interactions=[DipoleDipoleSoftened(eps=0.008)],
)

result = chain.simulate(
    theta0, omega0,
    dt=1e-3,
    steps=6000,
)


Save results:

from visualise_magnetic_cradle import save_result_npz
save_result_npz(result, "run1.npz")

Visualisation
Static plots
python visualise_magnetic_cradle.py \
    --npz run1.npz \
    --L 0.25 \
    --d 0.05 \
    --plot

Animation (interactive)
python visualise_magnetic_cradle.py \
    --npz run1.npz \
    --L 0.25 \
    --d 0.05 \
    --animate

Save MP4
python visualise_magnetic_cradle.py \
    --npz run1.npz \
    --L 0.25 \
    --d 0.05 \
    --animate \
    --fps 60 \
    --save_mp4 cradle.mp4


Optional trail:

--trail 30
