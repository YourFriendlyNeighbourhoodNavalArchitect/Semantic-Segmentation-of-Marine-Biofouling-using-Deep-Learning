"""
Generate publication-quality training curves from trialLog JSON files.
Plots both Attention U-Net and Simple U-Net on the same figures.

Usage:
    python plot_training_curves.py
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 600,
})

# ============================================================
# EDIT THESE PATHS to match your project
# ============================================================
ATT_PATH = Path(r"D:\Orfeas\PhD\Research\Semantic-Segmentation-of-Marine-Biofouling-using-Deep-Learning\Marine 2025\Trained models\AttentionUNet\trialLog.json")
SMP_PATH = Path(r"D:\Orfeas\PhD\Research\Semantic-Segmentation-of-Marine-Biofouling-using-Deep-Learning\Marine 2025\Trained models\SimpleUNet\trialLog.json")
OUTPUT_DIR = "."
# ============================================================
 
 
def load_trial(path):
    with open(path) as f:
        data = json.load(f)["Trial 0"]
    epochs = sorted(data.keys(), key=lambda x: int(x.split()[-1]))
    result = {"epoch": []}
    keys = ["Loss", "Dice Coefficient", "Accuracy"]
    for prefix in ["train", "val"]:
        for k in keys:
            result[f"{prefix}_{k}"] = []
 
    for ep in epochs:
        e = data[ep]
        result["epoch"].append(int(ep.split()[-1]))
        for src, prefix in [("trainingMetrics", "train"), ("validationMetrics", "val")]:
            m = e[src]
            for k in keys:
                result[f"{prefix}_{k}"].append(m.get(k, np.nan))
 
    for k in result:
        result[k] = np.array(result[k])
    return result
 
 
def moving_average(data, window=15):
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    padded = np.pad(data, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(data)]
 
 
def setup_axis(ax, max_epoch, title, ylabel):
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ticks = [1] + list(range(25, int(max_epoch) + 1, 25))
    # Drop last regular tick if too close to final epoch
    if ticks[-1] != int(max_epoch) and int(max_epoch) - ticks[-1] < 15:
        ticks[-1] = int(max_epoch)
    elif ticks[-1] != int(max_epoch):
        ticks.append(int(max_epoch))
    ax.set_xticks(ticks)
 
 
def main():
    att = load_trial(ATT_PATH)
    smp = load_trial(SMP_PATH)
    max_epoch = max(att["epoch"].max(), smp["epoch"].max())
 
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(wspace=0.3)
 
    # === LEFT: Loss ===
    # Raw (faint)
    axL.plot(att["epoch"], att["train_Loss"], color="#3266ad", lw=0.5, alpha=0.25)
    axL.plot(att["epoch"], att["val_Loss"], color="#d85a30", lw=0.5, alpha=0.25)
    axL.plot(smp["epoch"], smp["train_Loss"], color="#1d9e75", lw=0.5, alpha=0.25)
    axL.plot(smp["epoch"], smp["val_Loss"], color="#9b59b6", lw=0.5, alpha=0.25)
    # Moving average (bold)
    axL.plot(att["epoch"], moving_average(att["train_Loss"]),
             color="#3266ad", lw=2, label="Att U-Net Train")
    axL.plot(att["epoch"], moving_average(att["val_Loss"]),
             color="#d85a30", lw=2, ls="--", label="Att U-Net Val")
    axL.plot(smp["epoch"], moving_average(smp["train_Loss"]),
             color="#1d9e75", lw=2, label="Simple U-Net Train")
    axL.plot(smp["epoch"], moving_average(smp["val_Loss"]),
             color="#9b59b6", lw=2, ls="--", label="Simple U-Net Val")
 
    setup_axis(axL, max_epoch, "Training and Validation Loss", "Loss")
    axL.legend(loc="upper right", framealpha=0.9, edgecolor="gray")
 
    # === RIGHT: Dice + Accuracy ===
    # Raw (faint)
    for data, clr_d, clr_a in [(att, "#3266ad", "#d85a30"), (smp, "#1d9e75", "#9b59b6")]:
        axR.plot(data["epoch"], data["val_Dice Coefficient"], color=clr_d, lw=0.5, alpha=0.25)
        axR.plot(data["epoch"], data["val_Accuracy"], color=clr_a, lw=0.5, alpha=0.25)
 
    # Moving average (bold)
    axR.plot(att["epoch"], moving_average(att["val_Dice Coefficient"]),
             color="#3266ad", lw=2, label="Att U-Net Dice")
    axR.plot(att["epoch"], moving_average(att["val_Accuracy"]),
             color="#d85a30", lw=2, ls="--", label="Att U-Net Accuracy")
    axR.plot(smp["epoch"], moving_average(smp["val_Dice Coefficient"]),
             color="#1d9e75", lw=2, label="Simple U-Net Dice")
    axR.plot(smp["epoch"], moving_average(smp["val_Accuracy"]),
             color="#9b59b6", lw=2, ls="--", label="Simple U-Net Accuracy")
 
    setup_axis(axR, max_epoch, "Dice Coefficient and Accuracy", "Score")
    axR.legend(loc="lower right", framealpha=0.9, edgecolor="gray")
 
    fig.savefig(f"{OUTPUT_DIR}/fig_metrics_visualization.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{OUTPUT_DIR}/fig_metrics_visualization.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved: fig_metrics_visualization.png + .pdf")
 
 
if __name__ == "__main__":
    main()