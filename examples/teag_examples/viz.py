"""Matplotlib helpers (Agg backend) for documentation figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


def save_corr_heatmap(corr: pd.DataFrame, title: str, dest: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    data = corr.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    return dest


def save_label_bars(
    counts_by_table: dict[str, dict[int, int]],
    title: str,
    dest: Path,
    label_names: dict[int, str],
) -> Path:
    tables = list(counts_by_table)
    keys = sorted({k for counts in counts_by_table.values() for k in counts})
    x = np.arange(len(tables))
    width = 0.8 / max(len(keys), 1)
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for i, k in enumerate(keys):
        vals = [counts_by_table[t].get(k, 0) for t in tables]
        ax.bar(x + (i - (len(keys) - 1) / 2) * width, vals, width, label=label_names.get(k, str(k)))
    ax.set_xticks(x)
    ax.set_xticklabels(tables, rotation=20, ha="right")
    ax.set_ylabel("sentences")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    return dest


def save_fusion_shapes(dest: Path) -> Path:
    """Schematic of concat fusion (not a trained net)."""
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        (0.3, 1.3, 2.2, 1.6, "pooler\nB×768"),
        (3.0, 1.3, 2.2, 1.6, "gaze Linear\nB×5 → B×16"),
        (5.8, 1.3, 2.2, 1.6, "concat\nB×784"),
        (8.6, 1.3, 2.8, 1.6, "drop + Linear\nB×3 logits"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(
            plt.Rectangle((x, y), w, h, fill=True, facecolor="#dbeafe", edgecolor="#1e3a5f", lw=1.5)
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
    for x0, x1 in ((2.5, 3.0), (5.2, 5.8), (8.0, 8.6)):
        ax.annotate("", xy=(x1, 2.1), xytext=(x0, 2.1), arrowprops=dict(arrowstyle="->", color="#1e3a5f"))
    ax.set_title("EyeTrackingModel fusion (sentence gaze, no token alignment)")
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    return dest
