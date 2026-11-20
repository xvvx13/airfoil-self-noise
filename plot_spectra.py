"""Reproduce the paired spectra used in the report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from study import RESULTS, ROOT, ANGLE, SPEED, CHORD, THICK, FREQ, LEVEL

data = pd.read_csv(RESULTS / "test_predictions.csv")
condition = data.loc[(data[SPEED] == 55.5) & (data[ANGLE] == 0)]
styles = [(LEVEL, "Measured", "#182d45", "o", "-"),
          ("random_forest", "Random forest", "#bd481f", "s", "--")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), sharey=True)
for ax, chord, panel in zip(axes, [condition[CHORD].max(), condition[CHORD].min()], ["a", "b"]):
    curve = condition[condition[CHORD] == chord].sort_values(FREQ)
    assert curve[THICK].nunique() == 1
    for column, label, color, marker, linestyle in styles:
        ax.plot(curve[FREQ] / 1000, curve[column], color=color, linestyle=linestyle,
                marker=marker, markersize=3.5, linewidth=1.5, label=label,
                markerfacecolor="white" if column == "random_forest" else color)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4, 8, 16], ["0.5", "1", "2", "4", "8", "16"])
    ax.set_xlim(0.43, 23)
    ax.set_ylim(108, 140)
    ax.set_yticks([110, 115, 120, 125, 130, 135, 140])
    ax.tick_params(which="minor", bottom=False)
    ax.grid(axis="y", color="#dde2e6", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.set_title(f"({panel})  {1000 * chord:g} mm chord", loc="left", weight="bold", fontsize=10)
    ax.set_xlabel("Frequency (kHz, log scale)")
axes[0].set_ylabel("Scaled sound pressure level (dB)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
           bbox_to_anchor=(0.53, 1), fontsize=9)
fig.subplots_adjust(left=0.09, right=0.982, bottom=0.2, top=0.76, wspace=0.15)
(ROOT / "report").mkdir(exist_ok=True)
fig.savefig(ROOT / "report" / "spectra.pdf", metadata={"Title":"Held-out NACA 0012 spectra"})
