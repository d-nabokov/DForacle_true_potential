#!/usr/bin/env python3
"""
Plot LDPC-oracle experiment results.

Usage
-----
python plot_ldpc_results.py results.txt
"""

import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

DIFF_START = "Difference data for batches: "
ORACLE_START = "Oracle calls data for batches: "


def parse_file(file_path: Path):
    """Extract numpy arrays `difference_data` and `oracle_calls_data`."""
    difference_data = oracle_calls_data = None

    with file_path.open("rt") as f:
        for line in f:
            if line.startswith(DIFF_START):
                # safer than eval()
                difference_data = np.array(
                    ast.literal_eval(line[len(DIFF_START) :].strip())
                )
            elif line.startswith(ORACLE_START):
                oracle_calls_data = np.array(
                    ast.literal_eval(line[len(ORACLE_START) :].strip())
                )

    if difference_data is None or oracle_calls_data is None:
        raise ValueError("Input file missing required lines.")
    return difference_data, oracle_calls_data


def build_graphs(
    difference_data: np.ndarray, oracle_calls_data: np.ndarray, distances: list
):
    """Generate the two requested matplotlib figures."""
    diff_mean = difference_data.mean(axis=0)
    oracle_mean = oracle_calls_data.mean(axis=0)

    # ---------- Graph 1 ----------
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(oracle_mean, diff_mean, marker="o", markersize=5, lw=2)
    ax1.set_xlabel("Average Number of Traces", fontsize=26)
    ax1.set_ylabel("Avg. Hamming Distance", fontsize=26)
    ax1.tick_params(axis="both", which="major", labelsize=20)
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8, integer=True))
    ax1.grid(True)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    success_full = (difference_data == 0).mean(axis=0)
    ax2.plot(
        oracle_mean, success_full, marker="o", label="dist = 0", markersize=5, lw=2
    )
    for dist in distances:
        success_le = (difference_data <= dist).mean(axis=0)
        ax2.plot(
            oracle_mean,
            success_le,
            marker="x",
            label=f"dist ≤ {dist}",
            markersize=6,
            lw=2,
        )

    ax2.set_xlabel("Average Number of Traces", fontsize=26)
    ax2.set_ylabel("Success Probability", fontsize=26)
    ax2.tick_params(axis="both", which="major", labelsize=20)
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_xlim(left=1800, right=(oracle_mean[-1] + 20))
    ax2.grid(True)
    ax2.legend(fontsize=26)

    return fig1, fig2


def main():
    parser = argparse.ArgumentParser(description="Plot LDPC experiment output.")
    parser.add_argument(
        "infile", type=Path, help="Path to the text file produced by the experiments"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save PNGs instead of (or in addition to) showing the figures",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        help="Prefix of stored files",
    )
    parser.add_argument(
        "--dist",
        type=int,
        nargs="+",
        default=[2],
        help="Plot a different line for each distance",
    )
    args = parser.parse_args()

    diff_data, oracle_data = parse_file(args.infile)
    fig1, fig2 = build_graphs(diff_data, oracle_data, args.dist)

    if args.save:
        if args.prefix:
            prefix = args.prefix + "_"
        else:
            prefix = ""
        fig1.savefig(f"{prefix}diff_vs_oracle.pdf", bbox_inches="tight")
        fig2.savefig(f"{prefix}success_vs_oracle.pdf", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
