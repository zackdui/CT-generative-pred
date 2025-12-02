#!/usr/bin/env python3
"""
plot_lr.py — Plot learning rate curves from a PyTorch Lightning CSV metrics file.

Usage (from CLI):
    python plot_lr.py /path/to/metrics.csv --out lr_curve.png           # save PNG, no display
    python plot_lr.py /path/to/metrics.csv                               # show interactively
    python plot_lr.py /path/to/metrics.csv --step-col global_step        # override step column
    python plot_lr.py /path/to/metrics.csv --no-show --out lr.png        # headless save

Usage (from Python):
    from plot_lr import plot_lr_from_metrics
    fig = plot_lr_from_metrics("logs/my_run/version_0/metrics.csv", out_path="lr_curve.png", show=False)

Notes:
 - Works with LearningRateMonitor and/or custom LR logs (e.g., self.log("lr/current", ...)).
 - With gradient accumulation, LR changes only on optimizer steps, so points may appear every K batches.
"""

from __future__ import annotations
import argparse
import re
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import os


def _infer_step_col(df: pd.DataFrame, override: Optional[str] = None) -> str:
    if override is not None:
        if override not in df.columns:
            raise ValueError(f"Requested step_col='{override}' not found. Available: {list(df.columns)}")
        return override
    # Common Lightning CSV step columns
    for cand in ("step", "global_step", "trainer/global_step", "epoch_step", "epoch"):
        if cand in df.columns:
            return cand
    # Best effort: first column containing 'step'
    for c in df.columns:
        if "step" in c.lower():
            return c
    raise ValueError("Could not infer a step column (looked for step/global_step/epoch). "
                     "Pass step_col explicitly.")


def _find_lr_columns(df: pd.DataFrame) -> Iterable[str]:
    cols = []
    # Match starts with 'lr', e.g., 'lr', 'lr/pg0', 'lr-AdamW-0', and common custom keys like 'lr/current'
    rx = re.compile(r"^(lr($|[\/\-].*))|(^lr_current$)|(^lr\/current$)", re.IGNORECASE)
    for c in df.columns:
        if rx.match(c):
            cols.append(c)
    return cols


def plot_lr_from_metrics(csv_path: str,
                         step_col: Optional[str] = None,
                         out_path: Optional[str] = None,
                         show: bool = True):
    """
    Plot learning-rate curves from a Lightning CSV metrics file.

    Args:
        csv_path: Path to metrics.csv
        step_col: Column to use for the x-axis (default tries 'step', then 'global_step', etc.).
        out_path: If provided, save the figure to this path (PNG, PDF, etc.).
        show: If True, display the plot window (use False on headless servers).

    Returns:
        matplotlib.figure.Figure
    """
    df = pd.read_csv(csv_path)

    # Find LR columns
    lr_cols = list(_find_lr_columns(df))
    if not lr_cols:
        raise RuntimeError("No learning rate columns found. "
                           "Make sure you enabled LearningRateMonitor(logging_interval='step') "
                           "or logged a key like 'lr' or 'lr/current'.")

    # Pick the step/global_step column
    step_key = _infer_step_col(df, step_col)

    # Keep only rows where at least one LR was logged
    df = df.dropna(subset=lr_cols, how="all")

    if df.empty:
        raise RuntimeError("After dropping rows without LR logs, nothing remained. "
                           "Did you set Trainer(log_every_n_steps=1)?")

    fig = plt.figure()
    ax = fig.gca()
    for c in lr_cols:
        sub = df[[step_key, c]].copy()
        sub[step_key] = pd.to_numeric(sub[step_key], errors="coerce")
        sub[c]       = pd.to_numeric(sub[c], errors="coerce")
        sub = sub.dropna(subset=[step_key, c]).sort_values(step_key)
        if not sub.empty:
            ax.plot(sub[step_key], sub[c], label=c)
    ax.set_xlabel(step_key)
    ax.set_ylabel("learning rate")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2e"))
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"[plot_lr.py] Saved LR plot to: {out_path}")
    else:
        folder = os.path.dirname(csv_path)
        path = os.path.join(folder, "lr_plot.png")
        fig.savefig(path, dpi=150)
        print(f"[plot_lr.py] Saved LR plot to: {path}")
    if show:
        plt.show()
    
    return fig

def plot_loss_from_metrics(csv_path: str, out_path: Optional[str] = None, show: bool = False):
    df = pd.read_csv(csv_path)

    # Find Loss Columns
    loss_cols = [c for c in df.columns if 'loss' in c.lower()]

    # Pick the step/global_step column
    step_key = _infer_step_col(df)

    df.dropna(subset=loss_cols, how="all")

    if df.empty:
        raise RuntimeError("After dropping rows without loss logs, nothing remained. "
                           "Did you set Trainer(log_every_n_steps=1)?")

    fig = plt.figure()
    ax = fig.gca()
    for c in loss_cols:
        sub = df[[step_key, c]].copy()
        sub[step_key] = pd.to_numeric(sub[step_key], errors="coerce")
        sub[c]       = pd.to_numeric(sub[c], errors="coerce")
        sub = sub.dropna(subset=[step_key, c]).sort_values(step_key)
        if not sub.empty:
            ax.plot(sub[step_key], sub[c], label=c)
    ax.set_xlabel(step_key)
    ax.set_ylabel("loss value")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2e"))
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"[plot_lr.py] Saved loss plot to: {out_path}")
    else:
        folder = os.path.dirname(csv_path)
        path = os.path.join(folder, "loss_plot.png")
        fig.savefig(path, dpi=150)
        print(f"[plot_lr.py] Saved loss plot to: {path}")
    if show:
        plt.show()
    
    return fig



def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot learning rate from Lightning CSV metrics.")
    p.add_argument("csv_path", type=str, help="Path to metrics.csv")
    p.add_argument("--step-col", type=str, default=None, help="Override step column (e.g., global_step)")
    p.add_argument("--out", type=str, default=None, help="Optional output image path (e.g., lr.png)")
    p.add_argument("--no-show", action="store_true", help="Do not display the plot window")
    return p


def main():
    args = _build_argparser().parse_args()
    plot_lr_from_metrics(args.csv_path,
                         step_col=args.step_col,
                         out_path=args.out,
                         show=not args.no_show)


if __name__ == "__main__":
    main()
