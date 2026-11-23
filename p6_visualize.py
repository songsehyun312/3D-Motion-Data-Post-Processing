import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p5_analysis import ground_per_frame, smooth, ROOT, FPS   # same functions as the analysis

# BlazePose skeleton connections
BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
         (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
         (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
         (0, 11), (0, 12)]
TRAIL = 12   # frames of hand/foot trail to show the jitter


def draw(ax, seq, i, title, ground):
    ax.cla()
    p = seq[i]
    # plot (x, depth, height): matplotlib's z axis is vertical
    X, Y, Z = p[:, 0], p[:, 2], p[:, 1]
    for a, b in BONES:
        ax.plot([X[a], X[b]], [Y[a], Y[b]], [Z[a], Z[b]], color="#006CB8", lw=2)
    ax.scatter(X, Y, Z, s=8, color="#C2410C")
    for j in (15, 16, 31, 32):                      # trails of hands and feet
        k = max(0, i - TRAIL)
        ax.plot(seq[k:i + 1, j, 0], seq[k:i + 1, j, 2], seq[k:i + 1, j, 1], color="#C2410C", lw=1, alpha=0.7)
    if ground:
        gx, gy = np.meshgrid([-1, 1], [-1, 1])
        ax.plot_surface(gx, gy, np.zeros_like(gx), alpha=0.15, color="gray")
    lo, hi = LIM
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(lo, hi)
    ax.set_box_aspect((2, 2, hi - lo))
    ax.set_title(title, fontsize=11)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.view_init(elev=12, azim=-70)


def main():
    global LIM
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--proc", default=os.path.join(ROOT, "data", "processed"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "analysis"))
    ap.add_argument("--step", type=int, default=3, help="use every N-th frame (default 3)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    raw = np.load(os.path.join(args.proc, f"{args.name}_raw.npy"))
    prov, _ = ground_per_frame(raw)
    sm = smooth(prov)
    sm[:, :, 1] = np.maximum(sm[:, :, 1], 0)
    # raw is centred on the hips, so shift it up only for display (so all panels share the same height range)
    raw_show = raw.copy()
    raw_show[:, :, 1] -= raw[:, :, 1].min()
    LIM = (0, float(prov[:, :, 1].max()) + 0.1)
    panels = [(raw_show, "Raw", False), (prov, "Provided", True), (sm, "Provided + smoothing", True)]

    idx = list(range(0, len(raw), args.step))
    fig = plt.figure(figsize=(15, 5.5))
    axes = [fig.add_subplot(1, 3, k + 1, projection="3d") for k in range(3)]

    def frame(n):
        i = idx[n]
        for ax, (s, t, g) in zip(axes, panels):
            draw(ax, s, i, t, g)
        fig.suptitle(f"{args.name}   t = {i / FPS:.1f} s", fontsize=12)

    # still image with a few frames for the report
    picks = [len(idx) // 4, len(idx) // 2, 3 * len(idx) // 4]
    fig2, ax2 = plt.subplots(3, 3, figsize=(12, 11), subplot_kw=dict(projection="3d"))
    for r_, n in enumerate(picks):
        for c_, (s, t, g) in enumerate(panels):
            draw(ax2[r_, c_], s, idx[n], t if r_ == 0 else "", g)
            if c_ == 0:
                ax2[r_, c_].text2D(0.02, 0.92, f"t = {idx[n] / FPS:.1f} s", transform=ax2[r_, c_].transAxes)
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.out, f"{args.name}_compare_frames.png"), dpi=130)
    plt.close(fig2)

    fps_out = FPS / args.step
    anim = animation.FuncAnimation(fig, frame, frames=len(idx))
    try:
        path = os.path.join(args.out, f"{args.name}_compare.mp4")
        anim.save(path, writer=animation.FFMpegWriter(fps=fps_out, bitrate=3000))
    except Exception as e:                           # no ffmpeg -> gif
        print("mp4 failed (", e, ") -> saving gif")
        path = os.path.join(args.out, f"{args.name}_compare.gif")
        anim.save(path, writer=animation.PillowWriter(fps=fps_out))
    print("saved", path)


if __name__ == "__main__":
    main()
