import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FPS = 60.0                      # same value that p1_2_post_processing.py passes to PoseViser
L_FOOT, R_FOOT = 31, 32         # BlazePose foot index joints
HIPS = (23, 24)
GROUPS = {"head": [0], "hands": [15, 16], "feet": [31, 32], "all joints": list(range(33))}
BONES = [(11, 13), (13, 15), (12, 14), (14, 16), (23, 25), (25, 27), (24, 26), (26, 28)]


# ----------------------------------------------------------------- processing variants
def ground_per_frame(seq):
    """The provided method: lowest foot -> y=0 in every frame, then clamp y>=0."""
    s = seq.copy()
    foot_y = np.minimum(s[:, L_FOOT, 1], s[:, R_FOOT, 1])
    s[:, :, 1] -= foot_y[:, None]
    n_clamped = int((s[:, :, 1] < 0).sum())
    s[:, :, 1] = np.maximum(s[:, :, 1], 0)
    return s, n_clamped


def ground_global(seq):
    """One constant offset for the whole clip (lowest foot over all frames -> 0)."""
    s = seq.copy()
    foot_y = np.minimum(s[:, L_FOOT, 1], s[:, R_FOOT, 1])
    s[:, :, 1] -= foot_y.min()
    s[:, :, 1] = np.maximum(s[:, :, 1], 0)
    return s


def sg_coeffs(window, poly):
    half = window // 2
    x = np.arange(-half, half + 1)
    A = np.vander(x, poly + 1, increasing=True)
    return np.linalg.pinv(A)[0]


def smooth(seq, window=15, poly=3):
    """Savitzky-Golay smoothing along time (numpy only)."""
    c = sg_coeffs(window, poly)
    half = window // 2
    pad = np.pad(seq, ((half, half), (0, 0), (0, 0)), mode="edge")
    out = np.zeros_like(seq)
    for j in range(window):
        out += c[j] * pad[j:j + len(seq)]
    return out


# ----------------------------------------------------------------- metrics
def metrics(seq, ref=None):
    T = len(seq)
    vel = np.diff(seq, axis=0) * FPS
    acc = np.diff(seq, 2, axis=0) * FPS ** 2
    acc_n = np.linalg.norm(acc, axis=2)
    spd = np.linalg.norm(vel, axis=2)
    r = {}
    for g, j in GROUPS.items():
        r[f"jitter_{g}"] = acc_n[:, j].mean()          # mean |acceleration| (m/s^2)
    r["accel_p99"] = np.percentile(acc_n, 99)
    r["speed_p99"] = np.percentile(spd, 99)

    cvs = []
    for a, b in BONES:
        d = np.linalg.norm(seq[:, a] - seq[:, b], axis=1)
        cvs.append(d.std() / d.mean() * 100)
    r["bone_length_cv_pct"] = float(np.mean(cvs))

    hip_h = seq[:, list(HIPS), 1].mean(1)
    r["hip_height_std_cm"] = hip_h.std() * 100
    r["hip_height_accel"] = np.abs(np.diff(hip_h, 2)).mean() * FPS ** 2

    low = np.where(seq[:, L_FOOT, 1] <= seq[:, R_FOOT, 1], 0, 1)[:-1]
    hv = np.linalg.norm(np.diff(seq[:, [L_FOOT, R_FOOT]][:, :, [0, 2]], axis=0), axis=2) * FPS
    slide = hv[np.arange(T - 1), low]
    r["lowest_foot_slide_mean"] = slide.mean()
    r["lowest_foot_slide_p95"] = np.percentile(slide, 95)

    if ref is not None:
        r["rms_vs_provided_mm"] = np.sqrt(((seq - ref) ** 2).sum(2).mean()) * 1000
    return r


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--proc", default=os.path.join(ROOT, "data", "processed"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "analysis"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    raw = np.load(os.path.join(args.proc, f"{args.name}_raw.npy"))
    target = np.load(os.path.join(args.proc, f"{args.name}_target.npy"))
    T = len(raw)
    t = np.arange(T) / FPS
    print(f"[{args.name}] frames={T}  duration={T / FPS:.1f}s (fps={FPS:g} assumed)")

    v1, n_clamped = ground_per_frame(raw)
    print("reproduces provided target:", np.allclose(v1, target))
    print(f"joint-frames clamped to y=0: {n_clamped} / {T * 33} ({n_clamped / (T * 33) * 100:.2f}%)")

    v3 = smooth(v1)                                   # smooth the provided result ...
    v3[:, :, 1] = np.maximum(v3[:, :, 1], 0)          # ... and keep y >= 0
    variants = {
        "V0 raw (no ground fix)": raw,
        "V1 provided (per-frame ground)": v1,
        "V2 global ground offset": ground_global(raw),
        "V3 provided + SG smoothing": v3,
        "V4 SG smoothing -> per-frame ground": ground_per_frame(smooth(raw))[0],
    }

    table = {k: metrics(v, ref=v1) for k, v in variants.items()}
    keys = list(next(iter(table.values())).keys())

    # ---- print + csv
    print()
    print(f"{'metric':<26}" + "".join(f"{k.split()[0]:>10}" for k in table))
    for m in keys:
        print(f"{m:<26}" + "".join(f"{table[k].get(m, float('nan')):>10.2f}" for k in table))
    with open(os.path.join(args.out, f"{args.name}_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + list(table))
        for m in keys:
            w.writerow([m] + [f"{table[k].get(m, float('nan')):.4f}" for k in table])
        w.writerow(["clamped_joint_frames_pct", f"{n_clamped / (T * 33) * 100:.4f}"])

    # ---- figure 1: hip height implied by the foot correction
    hip1 = variants["V1 provided (per-frame ground)"][:, list(HIPS), 1].mean(1)
    hip4 = variants["V4 SG smoothing -> per-frame ground"][:, list(HIPS), 1].mean(1)
    acc = np.abs(np.diff(hip1, 2))
    c = int(np.argmax(np.convolve(acc, np.ones(int(FPS)), "same")))
    lo, hi = max(0, c - int(FPS)), min(T, c + int(FPS))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 1]})
    for a, sl in zip(ax, [slice(0, T), slice(lo, hi)]):
        a.plot(t[sl], hip1[sl], lw=1, label="provided (per-frame ground)")
        a.plot(t[sl], hip4[sl], lw=1.5, label="SG smoothing -> per-frame ground")
        a.set_xlabel("time (s)")
        a.set_ylabel("hip height (m)")
        a.grid(alpha=0.3)
    ax[0].set_title("Hip height created by the foot-based correction")
    ax[1].set_title("zoom: noisiest 2 s")
    ax[0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, f"{args.name}_fig1_hip_height.png"), dpi=200)
    plt.close()

    # ---- figure 2: jitter by body part
    names = ["V0 raw (no ground fix)", "V1 provided (per-frame ground)",
             "V3 provided + SG smoothing", "V4 SG smoothing -> per-frame ground"]
    labels = ["raw", "provided", "provided + smooth", "smooth -> ground"]
    groups = list(GROUPS)
    x = np.arange(len(groups))
    wdt = 0.2
    fig, a = plt.subplots(figsize=(9, 4))
    for i, (n, lb) in enumerate(zip(names, labels)):
        vals = [table[n][f"jitter_{g}"] for g in groups]
        a.bar(x + (i - 1.5) * wdt, vals, wdt, label=lb)
    a.set_xticks(x)
    a.set_xticklabels(groups)
    a.set_ylabel("mean |acceleration| (m/s²)")
    a.set_title("Jitter before/after processing (lower = smoother)")
    a.legend()
    a.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, f"{args.name}_fig2_jitter.png"), dpi=200)
    plt.close()

    # ---- figure 3: smoothing strength trade-off
    wins = [5, 9, 15, 21, 31, 45]
    pts = []
    for w_ in wins:
        s_ = ground_per_frame(smooth(raw, window=w_))[0]
        m_ = metrics(s_, ref=v1)
        pts.append((w_, m_["rms_vs_provided_mm"], m_["jitter_all joints"], m_["speed_p99"]))
    fig, a = plt.subplots(figsize=(6.5, 4.2))
    a.plot([p[1] for p in pts], [p[2] for p in pts], "o-")
    for w_, xx, yy, _ in pts:
        a.annotate(f"w={w_}", (xx, yy), textcoords="offset points", xytext=(5, 5), fontsize=9)
    a.plot([0], [table["V1 provided (per-frame ground)"]["jitter_all joints"]], "rs", label="provided (no smoothing)")
    a.set_xlabel("deviation from provided result (mm RMS)")
    a.set_ylabel("mean |acceleration|, all joints (m/s²)")
    a.set_title("Smoothing strength: less jitter vs. fidelity")
    a.legend()
    a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, f"{args.name}_fig3_smoothing_tradeoff.png"), dpi=200)
    plt.close()

    print("\nSG window sweep (window, RMS mm, jitter, speed_p99):")
    for p in pts:
        print("  ", p[0], round(p[1], 1), round(p[2], 1), round(p[3], 2))
    print("\nsaved to", args.out)


if __name__ == "__main__":
    main()
