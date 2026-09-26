"""
Generate a comparative bar chart from benchmark results.

Usage:
    python3 plot_results.py

Input:  results/benchmark_results.csv  (produced by run_benchmark.py)
Output: results/benchmark_comparison.png
"""

import os
import csv
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "results", "benchmark_results.csv")
OUT_PATH   = os.path.join(SCRIPT_DIR, "results", "benchmark_comparison.png")

TYPE_COLORS = {
    "none":                       "#9E9E9E",
    "rule-based":                 "#1F77B4",
    "optimization":               "#FF7F0E",
    "reinforcement":              "#7030A0",
    "supervisé (depuis oracle RL)":"#2CA02C",
    "hybride (RL → supervisé)":   "#D62728",
    "supervised (from LLM)":      "#17BECF",
}


def main():
    if not os.path.exists(CSV_PATH):
        print(f"[ERREUR] {CSV_PATH} introuvable — lance d'abord run_benchmark.py")
        return

    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    methods  = [r["method"]           for r in rows]
    tx       = [float(r["transmissions"])    for r in rows]
    drops    = [float(r["dropped_packets"])  for r in rows]
    dec_us   = [float(r["decision_us_avg"])  for r in rows]
    mem      = [float(r["memory_B"])         for r in rows]
    colors   = [TYPE_COLORS.get(r["learning_type"], "#888") for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Benchmark des méthodes de décision — Power Manager IoT (TVCC)",
        fontsize=13, fontweight="bold",
    )

    panels = [
        (axes[0, 0], tx,     "Transmissions réussies", "Qualité de service"),
        (axes[0, 1], drops,  "Paquets perdus",          "Pertes (moins = mieux)"),
        (axes[1, 0], dec_us, "Temps de décision (µs)",  "Coût computationnel"),
        (axes[1, 1], mem,    "Mémoire (octets)",         "Empreinte déploiement"),
    ]
    for ax, values, xlabel, title in panels:
        bars = ax.barh(methods, values, color=colors)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
        for bar, v in zip(bars, values):
            ax.text(v, bar.get_y() + bar.get_height() / 2,
                    f"  {int(v)}" if v > 1 else f"  {v:.1f}",
                    va="center", fontsize=8)
    axes[1, 1].set_xscale("log")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=t)
        for t, c in TYPE_COLORS.items()
        if t in [r["learning_type"] for r in rows]
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=4,
        bbox_to_anchor=(0.5, -0.02), fontsize=8,
        title="Type d'apprentissage",
    )

    plt.tight_layout(rect=(0, 0.05, 1, 0.97))
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"[OK] Figure → {OUT_PATH}")


if __name__ == "__main__":
    main()
