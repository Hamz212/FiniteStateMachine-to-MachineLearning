"""
MLP distilled from Q-learning.

A Q-learning agent is trained first (offline), then used as a labelling
oracle to generate a supervised dataset.  A compact 3->8->3 MLP is trained
on these labels with cross-entropy loss.

This two-stage pipeline (RL oracle → supervised distillation) produces a
model that is both small enough for deployment and more flexible than the
tabular Q-table at inference time (continuous inputs, no binning).
"""

import numpy as np
from decision.base import DecisionMethod
from decision.q_learning import QLearning


def _relu(x):
    return np.maximum(0, x)


class MLPDistilled(DecisionMethod):
    name = "MLP distillé (Q-learning)"
    description = "MLP trained by supervised distillation from a Q-learning oracle."
    learning_type = "hybride (RL → supervisé)"

    HIDDEN = 8

    def __init__(self):
        rng = np.random.default_rng(0)
        self.W1 = rng.normal(0, 0.3, (3, self.HIDDEN)).astype(np.float32)
        self.b1 = np.zeros(self.HIDDEN, dtype=np.float32)
        self.W2 = rng.normal(0, 0.3, (self.HIDDEN, 3)).astype(np.float32)
        self.b2 = np.zeros(3, dtype=np.float32)

    @staticmethod
    def _normalize(state):
        return np.array([
            state["theta"]  / 50.0,
            (state["v_batt"] - 3.2) / 0.5,
            (state["v_sc"]   - 2.5) / 0.5,
        ], dtype=np.float32)

    def _logits(self, x):
        h = _relu(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def decide(self, state):
        x = self._normalize(state)
        return int(np.argmax(self._logits(x)))

    def memory_bytes(self):
        return (self.W1.nbytes + self.b1.nbytes
                + self.W2.nbytes + self.b2.nbytes)

    def train(self, env_factory):
        # Train oracle
        oracle = QLearning()
        oracle.train(env_factory)

        # Generate labelled dataset from oracle
        rng = np.random.default_rng(1)
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
            X[i] = self._normalize(st)
            y[i] = oracle.decide(st)

        # SGD training (cross-entropy)
        lr, EPOCHS, BATCH = 0.05, 80, 128
        for _ in range(EPOCHS):
            perm = rng.permutation(N)
            for i in range(0, N, BATCH):
                idx = perm[i: i + BATCH]
                xb, yb = X[idx], y[idx]
                h = _relu(xb @ self.W1 + self.b1)
                logits = h @ self.W2 + self.b2
                ez = np.exp(logits - logits.max(1, keepdims=True))
                p  = ez / ez.sum(1, keepdims=True)
                t  = np.zeros_like(p)
                t[np.arange(len(yb)), yb] = 1.0
                dl = (p - t) / len(yb)
                dW2 = h.T @ dl;   db2 = dl.sum(0)
                dh  = dl @ self.W2.T;  dh[h <= 0] = 0
                dW1 = xb.T @ dh;  db1 = dh.sum(0)
                self.W2 -= lr * dW2;  self.b2 -= lr * db2
                self.W1 -= lr * dW1;  self.b1 -= lr * db1
