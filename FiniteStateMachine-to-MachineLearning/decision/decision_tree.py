"""
Decision tree distilled from a Q-learning oracle.

The tree is built by recursive binary splitting (CART-like) on samples
labelled by a trained Q-table. Maximum depth is capped at 6 to keep the
model interpretable and deployable on the target node.
"""

import numpy as np
from decision.base import DecisionMethod
from decision.q_learning import QLearning


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, feature=None, threshold=None,
                 left=None, right=None, value=None):
        self.feature   = feature
        self.threshold = threshold
        self.left      = left
        self.right     = right
        self.value     = value


def _gini(y):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=3) / len(y)
    return 1.0 - (p ** 2).sum()


class DecisionTreePolicy(DecisionMethod):
    name = "Arbre de décision"
    description = "Decision tree trained by distillation from a Q-learning oracle."
    learning_type = "supervisé (depuis oracle RL)"

    MAX_DEPTH = 6

    def __init__(self):
        self.tree = None
        self._n_nodes = 0

    @staticmethod
    def _features(state):
        return np.array(
            [state["theta"], state["v_batt"], state["v_sc"]],
            dtype=np.float32,
        )

    def decide(self, state):
        x = self._features(state)
        node = self.tree
        while node.value is None:
            node = node.left if x[node.feature] < node.threshold else node.right
        return int(node.value)

    def memory_bytes(self):
        return self._n_nodes * 6   # (feature, threshold, left_ptr) ~ 6 bytes

    def _build(self, X, y, depth):
        majority = int(np.bincount(y, minlength=3).argmax()) if len(y) else 0
        self._n_nodes += 1
        if len(y) == 0 or depth >= self.MAX_DEPTH or len(np.unique(y)) == 1:
            return _Node(value=majority)

        best_gini, best_feat, best_thr, best_mask = 1.0, 0, 0.0, None
        for f in range(3):
            for q in np.linspace(0.1, 0.9, 9):
                thr  = float(np.quantile(X[:, f], q))
                mask = X[:, f] < thr
                if mask.sum() == 0 or mask.sum() == len(y):
                    continue
                w = mask.sum() / len(y)
                g = w * _gini(y[mask]) + (1 - w) * _gini(y[~mask])
                if g < best_gini:
                    best_gini, best_feat, best_thr, best_mask = g, f, thr, mask

        if best_mask is None:
            return _Node(value=majority)
        return _Node(
            feature=best_feat, threshold=best_thr,
            left=self._build(X[best_mask],  y[best_mask],  depth + 1),
            right=self._build(X[~best_mask], y[~best_mask], depth + 1),
        )

    def train(self, env_factory):
        oracle = QLearning()
        oracle.train(env_factory)

        rng = np.random.default_rng(2)
        N = 20_000
        X = np.zeros((N, 3), dtype=np.float32)
        y = np.zeros(N, dtype=np.int32)
        for i in range(N):
            st = {
                "theta":           rng.uniform(-40, 85),
                "v_batt":          rng.uniform(2.8, 3.7),
                "v_sc":            rng.uniform(1.8, 3.3),
                "packets_pending": 0,
            }
            X[i] = np.array(
                [st["theta"], st["v_batt"], st["v_sc"]], dtype=np.float32
            )
            y[i] = oracle.decide(st)

        self._n_nodes = 0
        self.tree = self._build(X, y, depth=0)
