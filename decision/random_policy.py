"""
Random policy — sanity-check baseline.
Any reasonable method should outperform uniform random action selection.
"""

import numpy as np
from decision.base import DecisionMethod


class RandomPolicy(DecisionMethod):
    name = "Random Policy"
    description = "Uniform random action. Sanity check only."
    learning_type = "none"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def decide(self, state):
        return int(self.rng.integers(0, 3))

    def memory_bytes(self):
        return 4
