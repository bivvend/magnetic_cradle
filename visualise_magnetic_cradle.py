"""
visualise_magnetic_cradle.py

Visualiser for PlanarPendulumChain simulation results.

Assumes your simulator returned a dict like:
  result = {"t": (T,), "theta": (T,N), "omega": (T,N)}

This script supports:
1) Static plots (theta vs time, omega vs time)
2) Animation in x-y plane (bob positions vs time), using the same geometry
   as your PlanarPendulumChain: pivots at (i*d, 0), bob at:
       x = i*d + L*sin(theta)
       y = -L*cos(theta)

Usage examples:

A) If you run this as a script and load a saved .npz:
   python visualise_magnetic_cradle.py --npz run1.npz --L 0.25 --d 0.05 --animate
   python ./visualise_magnetic_cradle.py --npz run1.npz --L 0.25 --d 0.05 --plot --animate --trail 30

B) Or import and call functions directly from another script/notebook.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


@dataclass
class Geometry:
    L: float  # pendulum length
    d: float  # pivot spacing


def positions_from_theta(theta: np.ndarray, geom: Geometry) -> np.ndarray:
    """
    Convert theta time series to bob positions.

    theta: shape (T, N)
    returns positions: shape (T, N, 2) for (x,y) in the plane
    """
    theta = np.asarray(theta, dtype=float)
    T, N = theta.shape
    piv_x = np.arange(N, dtype=float) * geom.d

    x = piv_x[None, :] + geom.L * np.sin(theta)
    y = -geom.L * np.cos(theta)
    return np.stack([x, y], axis=-1)


def plot_timeseries(result: Dict[str, Any], show: bool = True, save: Optional[str] = None) -> None:
    """
    Two-panel plot: theta(t) and omega(t).
    """
    t = np.asarray(result["t"], dtype=float)
    theta = np.asarray(result["theta"], dtype=float)
    omega = np.asarray(result["omega"], dtype=float)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    ax1, ax2 = axes

    ax1.plot(t, theta)
    ax1.set_ylabel("theta (rad)")
    ax1.set_title("Angles vs time")

    ax2.plot(t, omega)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("omega (rad/s)")
    ax2.set_title("Angular velocities vs time")

    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def animate_cradle(
    result: Dict[str, Any],
    geom: Geometry,
    fps: int = 60,
    trail: int = 0,
    show: bool = True,
    save_mp4: Optional[str] = None,
) -> None:
    """
    Animate bobs (and rods) in the x-y plane.

    Parameters:
      fps: playback frame rate
      trail: number of previous frames to show as a faint trail (0 = off)
      save_mp4: if provided, attempts to save mp4 (needs ffmpeg installed)
    """
    t = np.asarray(result["t"], dtype=float)
    theta = np.asarray(result["theta"], dtype=float)

    pos = positions_from_theta(theta, geom)  # (T,N,2)
    Tn, N, _ = pos.shape

    piv = np.zeros((N, 2), dtype=float)
    piv[:, 0] = np.arange(N) * geom.d
    piv[:, 1] = 0.0

    # Choose animation stride so the playback is close to requested fps
    if len(t) >= 2:
        dt_sim = float(np.median(np.diff(t)))
        stride = max(1, int(round((1.0 / fps) / dt_sim)))
    else:
        stride = 1

    frame_ids = np.arange(0, Tn, stride, dtype=int)

    # --- Set plot limits to show full pendulum height ---
    x_all = pos[:, :, 0]
    y_all = pos[:, :, 1]

    # Horizontal span: bobs + pivots
    x_min = min(x_all.min(), piv[:, 0].min())
    x_max = max(x_all.max(), piv[:, 0].max())

    # Vertical span: always include pivots (y=0) and full length L
    y_top = 0.0
    y_bottom = -geom.L

    # Add a little breathing room
    pad_x = 0.15 * max(geom.d * (N - 1), geom.L)
    pad_y = 0.15 * geom.L

    x_min -= pad_x
    x_max += pad_x
    y_top += pad_y
    y_bottom -= pad_y

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_bottom, y_top)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Magnetic cradle (planar pendula)")
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)


    # Draw pivots
    ax.scatter(piv[:, 0], piv[:, 1], s=25)

    # One Line2D for rods (polyline segments)
    rods, = ax.plot([], [], lw=1)

    # One Line2D for bobs
    bobs, = ax.plot([], [], marker="o", linestyle="None", markersize=12)

    # Optional trail per bob: use a single line collection style by plotting
    # each bob's trail as a separate line (simple + readable).
    trail_lines = []
    if trail > 0:
        for _ in range(N):
            ln, = ax.plot([], [], lw=1, alpha=0.3)
            trail_lines.append(ln)

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def init():
        rods.set_data([], [])
        bobs.set_data([], [])
        time_text.set_text("")
        for ln in trail_lines:
            ln.set_data([], [])
        return (rods, bobs, time_text, *trail_lines)

    def update(frame_idx: int):
        k = frame_ids[frame_idx]
        xy = pos[k]  # (N,2)

        # Rod segments: interleave pivot and bob points with NaN separators
        xs = []
        ys = []
        for i in range(N):
            xs.extend([piv[i, 0], xy[i, 0], np.nan])
            ys.extend([piv[i, 1], xy[i, 1], np.nan])
        rods.set_data(xs, ys)

        # Bobs
        bobs.set_data(xy[:, 0], xy[:, 1])

        # Trails
        if trail > 0:
            start = max(0, k - trail * stride)
            window = pos[start : k + 1 : stride]  # (W,N,2)
            for i, ln in enumerate(trail_lines):
                ln.set_data(window[:, i, 0], window[:, i, 1])

        time_text.set_text(f"t = {t[k]:.3f} s")
        return (rods, bobs, time_text, *trail_lines)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_ids),
        init_func=init,
        interval=1000 / fps,
        blit=True,
    )

    if save_mp4:
        anim.save(save_mp4, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def save_result_npz(result: Dict[str, Any], path: str) -> None:
    """
    Convenience: save a result dict as .npz.
    """
    np.savez_compressed(path, t=result["t"], theta=result["theta"], omega=result["omega"])


def load_result_npz(path: str) -> Dict[str, Any]:
    """
    Load a .npz result file back into the result dict format.
    """
    data = np.load(path)
    return {"t": data["t"], "theta": data["theta"], "omega": data["omega"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True, help="Path to .npz results file")
    parser.add_argument("--L", type=float, required=True, help="Pendulum length (m)")
    parser.add_argument("--d", type=float, required=True, help="Pivot spacing (m)")
    parser.add_argument("--fps", type=int, default=60, help="Animation fps")
    parser.add_argument("--trail", type=int, default=0, help="Trail length in frames (0=off)")
    parser.add_argument("--plot", action="store_true", help="Plot theta/omega timeseries")
    parser.add_argument("--animate", action="store_true", help="Animate the cradle")
    parser.add_argument("--save_plot", type=str, default=None, help="Save plot to file (png)")
    parser.add_argument("--save_mp4", type=str, default=None, help="Save animation to mp4 (needs ffmpeg)")
    args = parser.parse_args()

    result = load_result_npz(args.npz)
    geom = Geometry(L=args.L, d=args.d)

    if args.plot:
        plot_timeseries(result, show=True, save=args.save_plot)

    if args.animate:
        animate_cradle(
            result,
            geom=geom,
            fps=args.fps,
            trail=args.trail,
            show=True,
            save_mp4=args.save_mp4,
        )

    if not args.plot and not args.animate:
        # default behaviour: do both
        plot_timeseries(result, show=True, save=args.save_plot)
        animate_cradle(result, geom=geom, fps=args.fps, trail=args.trail, show=True, save_mp4=args.save_mp4)


if __name__ == "__main__":
    main()


