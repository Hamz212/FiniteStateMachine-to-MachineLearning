"""
Benchmark of decision methods for the IoT node power manager.

Runs each method on the same TVCC scenario (fixed seeds), collects
performance metrics, prints a summary table, and saves results to CSV.

Usage:
    python3 run_benchmark.py

Results:
    results/benchmark_results.csv
    (use plot_results.py to generate the comparison figure)
"""

import os
import sys
import csv
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from environment import IoTNodeEnv
from decision.fsm_baseline   import FSMBaseline
from decision.fsm_optimized  import FSMOptimized
from decision.random_policy  import RandomPolicy
from decision.q_learning     import QLearning
from decision.mlp_distilled  import MLPDistilled
from decision.llm_distilled  import LLMDistilled
from decision.decision_tree  import DecisionTreePolicy
from decision.fuzzy_logic    import FuzzyLogic

EPISODE_MINUTES = 60 * 24 * 7   # 1 week
N_EVAL_SEEDS   = 5


def env_factory(seed=42):
    return IoTNodeEnv(seed=seed, episode_minutes=EPISODE_MINUTES)


def run_episode(method, env):
    state = env.get_state()
    done  = False
    times = []
    while not done:
        t0 = time.perf_counter()
        a  = method.decide(state)
        times.append(time.perf_counter() - t0)
        state, done = env.step(a)
    return times


def evaluate(method):
    acc = {k: [] for k in [
        "autonomy_days", "failures", "transmissions",
        "dropped_packets", "energy_J", "decision_us_avg",
    ]}
    for s in range(N_EVAL_SEEDS):
        env   = IoTNodeEnv(seed=1000 + s, episode_minutes=EPISODE_MINUTES)
        times = run_episode(method, env)
        acc["autonomy_days"].append(env.autonomy_days())
        acc["failures"].append(env.failures)
        acc["transmissions"].append(env.transmissions)
        acc["dropped_packets"].append(env.dropped_packets)
        acc["energy_J"].append(env.total_energy_consumed)
        acc["decision_us_avg"].append(np.mean(times) * 1e6)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    methods = [
        RandomPolicy(seed=7),
        FSMBaseline(),
        FSMOptimized(),
        FuzzyLogic(),
        QLearning(),
        DecisionTreePolicy(),
        MLPDistilled(),
        LLMDistilled(),
    ]

    header_cols = [
        ("method",           24, "s"),
        ("learning_type",    26, "s"),
        ("autonomy_days",     9, ".2f"),
        ("failures",          7, ".1f"),
        ("transmissions",     7, ".0f"),
        ("dropped_packets",   7, ".0f"),
        ("decision_us_avg",  11, ".1f"),
        ("memory_B",          8, "d"),
        ("train_s",           8, ".1f"),
    ]
    short = {
        "method":           "Méthode",
        "learning_type":    "Apprentissage",
        "autonomy_days":    "Auton(j)",
        "failures":         "Éch.",
        "transmissions":    "TX",
        "dropped_packets":  "Perdus",
        "decision_us_avg":  "T_déc(µs)",
        "memory_B":         "Mém(B)",
        "train_s":          "Train(s)",
    }

    W = sum(w + 1 for _, w, _ in header_cols)
    print("=" * W)
    print(f"BENCHMARK — {len(methods)} méthodes — scénario TVCC 1 semaine "
          f"× {N_EVAL_SEEDS} graines")
    print("=" * W)

    all_results = []
    for m in methods:
        print(f"\n► {m.name}  [{m.learning_type}]")
        t0 = time.perf_counter()
        m.train(env_factory=env_factory)
        train_s = time.perf_counter() - t0
        metrics = evaluate(m)
        metrics["method"]        = m.name
        metrics["learning_type"] = m.learning_type
        metrics["train_s"]       = train_s
        metrics["memory_B"]      = m.memory_bytes()
        all_results.append(metrics)
        print(
            f"  autonomie={metrics['autonomy_days']:.2f}j  "
            f"échecs={metrics['failures']:.0f}  "
            f"TX={metrics['transmissions']:.0f}  "
            f"perdus={metrics['dropped_packets']:.0f}  "
            f"décision={metrics['decision_us_avg']:.1f}µs  "
            f"mém={metrics['memory_B']}B  "
            f"train={train_s:.1f}s"
        )

    # Summary table
    print("\n" + "=" * W)
    print("RÉSULTATS")
    print("=" * W)
    line = "".join(f"{short[col]:<{w}} " for col, w, _ in header_cols)
    print(line)
    print("-" * W)
    for r in all_results:
        parts = []
        for col, w, fmt in header_cols:
            v = r[col]
            if fmt == "s":
                parts.append(f"{str(v)[:w]:<{w}}")
            elif fmt == "d":
                parts.append(f"{int(v):<{w}d}")
            else:
                parts.append(f"{v:<{w}{fmt}}")
        print(" ".join(parts))
    print("=" * W)

    # Save CSV
    os.makedirs(os.path.join(SCRIPT_DIR, "results"), exist_ok=True)
    csv_path = os.path.join(SCRIPT_DIR, "results", "benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[c for c, _, _ in header_cols])
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k, _, _ in header_cols})
    print(f"\n[OK] Résultats → {csv_path}")


if __name__ == "__main__":
    main()
