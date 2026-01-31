"""
magnetic_cradle.py

Planar pendulum chain with 1 DOF per bob (theta), but 3D dipole–dipole forces.
Magnets are rigidly mounted so they *can't spin independently*; dipole direction is tangential.

Design goals:
- Class-based + extendable interaction model (plug-in forces)
- Stable integrator (RK4) for stiff-ish interactions
- Clear separation between geometry, interactions, and time stepping
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Optional, Tuple
import numpy as np
import csv  
from visualise_magnetic_cradle import save_result_npz 


# ----------------------------
# Interaction plugin interface
# ----------------------------

class Interaction(Protocol):
    """Compute Cartesian forces on each bob, given positions and per-bob metadata."""
    def forces(
        self,
        positions: np.ndarray,   # shape (N, 3)
        dipoles: np.ndarray,     # shape (N, 3)  (may be unused by other interactions)
    ) -> np.ndarray:             # shape (N, 3)
        ...


# ---------------------------------------
# Dipole-dipole interaction with softening
# ---------------------------------------

@dataclass
class DipoleDipoleSoftened:
    """
    Softened point dipole–dipole interaction in 3D.

    Uses:
      F_{i<-j} = (3 μ0 / (4π r^4)) * [ (mi·rhat)mj + (mj·rhat)mi + (mi·mj)rhat - 5(mi·rhat)(mj·rhat)rhat ]

    with softening:
      r_eps = sqrt(|R|^2 + eps^2)
      rhat  = R / r_eps     (regularised; not strictly unit when eps>0, intentionally)
      r^4   = r_eps^4
    """
    mu0_over_4pi: float = 1e-7   # μ0/(4π) in SI
    eps: float = 1e-3            # softening length scale (m)
    pair_cutoff: Optional[float] = None  # optional cutoff (m); None => all pairs

    def forces(self, positions: np.ndarray, dipoles: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=float)
        dipoles = np.asarray(dipoles, dtype=float)
        N = positions.shape[0]
        F = np.zeros((N, 3), dtype=float)

        # Pairwise O(N^2) loop; fine for cradle-sized N.
        for i in range(N):
            ri = positions[i]
            mi = dipoles[i]
            for j in range(i + 1, N):
                R = ri - positions[j] #Modified 
                r2 = float(np.dot(R, R))
                if self.pair_cutoff is not None and r2 > self.pair_cutoff * self.pair_cutoff:
                    continue

                r_eps = np.sqrt(r2 + self.eps * self.eps)
                rhat = R / r_eps
                mj = dipoles[j]

                mi_r = float(np.dot(mi, rhat))
                mj_r = float(np.dot(mj, rhat))
                mi_mj = float(np.dot(mi, mj))

                pref = 3.0 * self.mu0_over_4pi / (r_eps ** 4)

                # Vector bracket term:
                bracket = (mi_r * mj) + (mj_r * mi) + (mi_mj * rhat) - (5.0 * mi_r * mj_r * rhat)
                Fij = pref * bracket

                # Action-reaction:
                F[i] += Fij
                F[j] -= Fij

        return F


# -------------------------------------------------
# (Optional) Example: nearest-neighbour-only wrapper
# -------------------------------------------------

@dataclass
class NearestNeighbourOnly:
    """
    Wraps another interaction but only applies it to immediate neighbours (i,i+1).
    Useful if you want more "cradle-like" impulse transfer as a controlled approximation.
    """
    base: Interaction

    def forces(self, positions: np.ndarray, dipoles: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=float)
        dipoles = np.asarray(dipoles, dtype=float)
        N = positions.shape[0]
        F = np.zeros((N, 3), dtype=float)

        # Only pairs (i, i+1)
        for i in range(N - 1):
            # Compute force for just these two by calling a tiny local version.
            # If base is DipoleDipoleSoftened, we can compute directly for speed.
            # Otherwise, fallback to a generic call on a 2-body system.
            if isinstance(self.base, DipoleDipoleSoftened):
                dd = self.base
                R = positions[i + 1] - positions[i]
                r2 = float(np.dot(R, R))
                r_eps = np.sqrt(r2 + dd.eps * dd.eps)
                rhat = R / r_eps
                mi = dipoles[i]
                mj = dipoles[i + 1]
                mi_r = float(np.dot(mi, rhat))
                mj_r = float(np.dot(mj, rhat))
                mi_mj = float(np.dot(mi, mj))
                pref = 3.0 * dd.mu0_over_4pi / (r_eps ** 4)
                bracket = (mi_r * mj) + (mj_r * mi) + (mi_mj * rhat) - (5.0 * mi_r * mj_r * rhat)
                Fij = pref * bracket
            else:
                local_pos = np.vstack([positions[i], positions[i + 1]])
                local_dip = np.vstack([dipoles[i], dipoles[i + 1]])
                local_F = self.base.forces(local_pos, local_dip)
                Fij = local_F[0]  # force on first due to second (action-reaction enforced by base)

            F[i] += Fij
            F[i + 1] -= Fij

        return F


# -----------------------------
# Pendulum chain simulator core
# -----------------------------

@dataclass
class PendulumChainParams:
    N: int
    m: float          # bob mass (kg)
    L: float          # pendulum length (m)
    d: float          # pivot spacing (m)
    g: float = 9.81   # m/s^2
    pivot_damping_c: float = 0.0  # torque damping coefficient in equation: -c*omega (N·m·s)


class PlanarPendulumChain:
    """
    1 DOF per bob: theta_i. Motion constrained to x-y plane.
    Positions are 3D vectors (z=0), so 3D dipole forces work naturally.

    Equations:
      r_i(theta) = p_i + [ L sinθ, -L cosθ, 0 ]
      t_hat_i    = [ cosθ,  sinθ, 0 ]         (unit tangent direction)
      dipole     = m_d * t_hat_i              (tangential)
      theta_ddot = -(g/L) sinθ + (1/(mL)) (F_mag · t_hat) - (c/(mL^2)) omega
    """

    def __init__(
        self,
        params: PendulumChainParams,
        dipole_magnitude: float,        # A·m^2
        interactions: Optional[List[Interaction]] = None,
        dipole_signs: Optional[np.ndarray] = None,
    ):
        self.p = params
        self.m_d = float(dipole_magnitude)
        self.interactions: List[Interaction] = interactions or []

        # Fixed pivot positions along x-axis
        self.pivots = np.zeros((self.p.N, 3), dtype=float)
        for i in range(self.p.N):
            self.pivots[i] = np.array([i * self.p.d, 0.0, 0.0], dtype=float)

        if dipole_signs is None:
            # Default: alternating +1, -1, +1, -1, ...
            self.dipole_signs = np.array([1.0 if (i % 2 == 0) else -1.0 for i in range(self.p.N)])
        else:
            dipole_signs = np.asarray(dipole_signs, dtype=float)
            if dipole_signs.shape != (self.p.N,):
                raise ValueError("dipole_signs must have shape (N,)")
            self.dipole_signs = dipole_signs

    # --- Geometry helpers ---

    def positions(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        x = self.pivots[:, 0] + self.p.L * np.sin(theta)
        y = self.pivots[:, 1] - self.p.L * np.cos(theta)
        z = np.zeros_like(theta)
        return np.column_stack([x, y, z])

    def t_hat(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        return np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])

    def dipoles(self, theta: np.ndarray) -> np.ndarray:
        # Tangential unit vectors (N,3)
        th = self.t_hat(theta)
        # Apply alternating sign per bob
        return (self.dipole_signs[:, None] * self.m_d) * th

    # --- Dynamics ---

    def magnetic_forces(self, theta: np.ndarray) -> np.ndarray:
        r = self.positions(theta)
        mvec = self.dipoles(theta)
        F = np.zeros_like(r)
        for inter in self.interactions:
            F += inter.forces(r, mvec)
        return F

    def accel(self, theta: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """
        Returns theta_ddot given theta and omega.
        """
        theta = np.asarray(theta, dtype=float)
        omega = np.asarray(omega, dtype=float)

        # Gravity term
        a = -(self.p.g / self.p.L) * np.sin(theta)

        # Magnetic term
        if self.interactions:
            Fmag = self.magnetic_forces(theta)
            t_hat = self.t_hat(theta)
            # (F · t_hat) / (m L)
            a += (np.einsum("ij,ij->i", Fmag, t_hat)) / (self.p.m * self.p.L)

        # Pivot damping: -(c/(m L^2)) * omega
        if self.p.pivot_damping_c != 0.0:
            a += -(self.p.pivot_damping_c / (self.p.m * self.p.L * self.p.L)) * omega

        return a

    # --- Time integration (RK4) ---

    def step_rk4(self, theta: np.ndarray, omega: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        One RK4 step for the coupled system:
          dtheta/dt = omega
          domega/dt = accel(theta, omega)
        """
        theta = np.asarray(theta, dtype=float)
        omega = np.asarray(omega, dtype=float)
        dt = float(dt)

        def f(th: np.ndarray, om: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return om, self.accel(th, om)

        k1_th, k1_om = f(theta, omega)
        k2_th, k2_om = f(theta + 0.5 * dt * k1_th, omega + 0.5 * dt * k1_om)
        k3_th, k3_om = f(theta + 0.5 * dt * k2_th, omega + 0.5 * dt * k2_om)
        k4_th, k4_om = f(theta + dt * k3_th, omega + dt * k3_om)

        theta_next = theta + (dt / 6.0) * (k1_th + 2 * k2_th + 2 * k3_th + k4_th)
        omega_next = omega + (dt / 6.0) * (k1_om + 2 * k2_om + 2 * k3_om + k4_om)

        return theta_next, omega_next

    def simulate(
        self,
        theta0: np.ndarray,
        omega0: np.ndarray,
        dt: float,
        steps: int,
        record_every: int = 1,
    ) -> dict:
        """
        Run a simulation and return a dict with time series.
        """
        theta = np.array(theta0, dtype=float).copy()
        omega = np.array(omega0, dtype=float).copy()

        out_t = []
        out_theta = []
        out_omega = []

        t = 0.0
        step_num = 0
        for s in range(steps):
            step_num +=1
            if s % record_every == 0:
                out_t.append(t)
                out_theta.append(theta.copy())
                out_omega.append(omega.copy())
                print(f"{int(100 * step_num / steps)} % Done")

            theta, omega = self.step_rk4(theta, omega, dt)
            t += dt

        return {
            "t": np.array(out_t),
            "theta": np.vstack(out_theta),   # shape (T, N)
            "omega": np.vstack(out_omega),   # shape (T, N)
        }



if __name__ == "__main__":
    # --- Parameters (tweak these) ---
    N = 3
    params = PendulumChainParams(
        N=N,
        m=0.02,          # 20 g bobs
        L=0.25,          # 25 cm pendulum
        d=0.05,          # 5 cm pivot spacing
        g=9.81,
        pivot_damping_c=1e-5,  # small damping; set 0 for conservative
    )

    # Dipole magnitude: depends hugely on magnet; start small and increase.
    dipole_magnitude = 0.3  # A·m^2 (toy starting value)

    # Interaction: softened dipole-dipole
    interaction = DipoleDipoleSoftened(eps=0.008)  # 8 mm softening ~ magnet size

    # Optionally wrap to nearest neighbours only:
    # interaction = NearestNeighbourOnly(DipoleDipoleSoftened(eps=0.008))

    chain = PlanarPendulumChain(
        params=params,
        dipole_magnitude=dipole_magnitude,
        interactions=[interaction],
        # optional: override signs if you want a different pattern
        dipole_signs=np.array([-1, +1, -1], dtype=float)
    )

    # Initial condition: lift leftmost bob
    theta0 = np.zeros(N)
    omega0 = np.zeros(N)
    theta0[0] = np.deg2rad(-20)

    dt = 1e-5
    T = 20
    steps = int(T / dt)

    result = chain.simulate(theta0, omega0, dt=dt, steps=steps, record_every=100)

    # Quick textual sanity output
    print("Simulated frames:", len(result["t"]))
    print("Final thetas (deg):", np.rad2deg(result["theta"][-1]))

    # Save results to CSV
    
    csv_filename = "simulation_results.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        header = ["time"]
        for i in range(N):
            header.append(f"theta_{i}_rad")
            header.append(f"omega_{i}_rad_per_s")
            
        writer.writerow(header)
        
        # Write data rows
        for idx, t in enumerate(result["t"]):
            row = [t]
            row.extend(result["theta"][idx])
            row.extend(result["omega"][idx])  
            writer.writerow(row)
    
    print(f"Results saved to {csv_filename}")

    
    save_result_npz(result, "run1.npz")
    
