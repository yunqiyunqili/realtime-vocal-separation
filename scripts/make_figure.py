"""Plot test vocals SDR against algorithmic latency from results/summary.csv -> results/latency_vs_sdr.png.

Colour (single-hue sequential ramp) and marker area both encode compute (GMAC per second of audio). Circles are
the causal Conv-TasNet tiers, squares the spectral / band-split comparison models; the dashed line is the offline,
non-causal MDX-Net reference (no algorithmic latency defined). C and S (both 11.6 ms) are drawn 0.3 ms apart."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]   # blue 250 -> 700
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def value(rows, section, model, metric, n="50"):
    for r in rows:
        if r["section"] == section and r["model"] == model and r["metric"] == metric and r["n_tracks"] in (n, ""):
            return float(r["value"])
    raise KeyError((section, model, metric))


def main():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "results", "summary.csv"))))
    pts = []
    for m, label in [("F23", "F23"), ("F12", "F12"), ("F6", "F6")]:
        pts.append((label, value(rows, "main", m, "algorithmic_latency"), value(rows, "main", m, "museval_vocals_sdr_median"),
                    value(rows, "main", m, "gmac_per_audio_second"), "o"))
    for m, label in [("C", "C  (FFT-512 band-split)"), ("S", "S  (asymmetric window)")]:
        pts.append((label, value(rows, "architecture", m, "algorithmic_latency"),
                    value(rows, "architecture", m, "museval_vocals_sdr_median"),
                    value(rows, "architecture", m, "gmac_per_audio_second"), "s"))
    mdx = value(rows, "reference", "MDX-Net Track A vocals (offline, non-causal)", "museval_vocals_sdr_median")

    cmap = LinearSegmentedColormap.from_list("blue", RAMP)
    norm = Normalize(1.0, 4.0)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    ax.axhline(mdx, color=INK2, lw=1.2, ls=(0, (5, 4)), zorder=1)
    ax.text(25.6, mdx - 0.18, f"MDX-Net offline reference (non-causal): {mdx:.2f} dB", ha="right", va="top", color=INK2, fontsize=9)
    offsets = {"F23": (-12, 14, "right"), "F12": (12, 10, "left"), "F6": (0, 24, "center"),
               "C  (FFT-512 band-split)": (-12, -14, "right"), "S  (asymmetric window)": (12, 8, "left")}
    dodge = {"C  (FFT-512 band-split)": -0.3, "S  (asymmetric window)": 0.3}   # C and S share 11.6 ms
    for label, lat, sdr, g, mk in sorted(pts, key=lambda p: -p[3]):   # small marks drawn on top
        lat = lat + dodge.get(label, 0.0)
        ax.scatter(lat, sdr, s=60 + 55 * g, c=[cmap(norm(g))], marker=mk, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.annotate(f"{label}\n{sdr:.2f} dB, {g:.2f} GMAC/s", (lat, sdr), xytext=offsets[label][:2], textcoords="offset points",
                    fontsize=8.5, color=INK, va="center", ha=offsets[label][2])
    ax.set_xlim(0, 26); ax.set_ylim(0, 10)
    ax.set_xticks([0, 5.8, 11.6, 23.2]); ax.set_xticklabels(["0", "5.8", "11.6", "23.2"])
    ax.set_xlabel("Algorithmic latency (ms)", color=INK2)
    ax.set_ylabel("Test vocals SDR (dB, museval v4, 50 tracks)", color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.set_title("Vocals SDR vs latency (colour and size = compute)", loc="left", color=INK, fontsize=11)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.04); cb.set_label("GMAC per second of audio", color=INK2)
    cb.outline.set_visible(False); cb.ax.tick_params(colors=INK2)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=INK2, label="causal Conv-TasNet"),
                       Line2D([], [], marker="s", ls="", color=INK2, label="spectral / band-split"),
                       Line2D([], [], ls=(0, (5, 4)), color=INK2, label="offline reference")],
              loc="lower right", frameon=False, fontsize=8.5)
    fig.text(0.01, 0.01, "C and S both have 11.6 ms latency; their markers are shifted by 0.3 ms for visibility.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = os.path.join(ROOT, "results", "latency_vs_sdr.png")
    fig.savefig(out, facecolor=SURFACE); print(out)


if __name__ == "__main__":
    main()
