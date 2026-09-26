"""
Tabular Q-learning for the IoT node power manager.

The state space is discretised into 4×3×3 = 36 bins (temperature × battery
voltage × SC voltage). The agent interacts with the simulator over 200
episodes with an epsilon-greedy policy and decaying exploration rate.
"""

import numpy as np
from decision.base import DecisionMethod


class QLearning(DecisionMethod):
    name = "Q-learning tabulaire"
    description = "Tabular RL with a 36-state Q-table and epsilon-greedy exploration."
    learning_type = "reinforcement"

    N_THETA  = 4
    N_VBATT  = 3
    N_VSC    = 3
    N_STATES = N_THETA * N_VBATT * N_VSC   # 36
    N_ACTIONS = 3

    def __init__(self):
        rng = np.random.default_rng(42)
        self.Q = rng.uniform(-0.01, 0.01, (self.N_STATES, self.N_ACTIONS))

    # ------------------------------------------------------------------
    # State discretisation
    # ------------------------------------------------------------------
    @staticmethod
    def _bin_theta(t):
        if t < -20:  return 0
        if t <   0:  return 1
        if t <  40:  return 2
        return 3

    @staticmethod
    def _bin_vbatt(v):
        if v < 3.2:  return 0
        if v < 3.5:  return 1
        return 2

    @staticmethod
    def _bin_vsc(v):
        if v < 2.2:  return 0
        if v < 2.9:  return 1
        return 2

    def _encode(self, state):
        a = self._bin_theta(state["theta"])
        b = self._bin_vbatt(state["v_batt"])
        c = self._bin_vsc(state["v_sc"])
        return (a * self.N_VBATT + b) * self.N_VSC + c

    # ------------------------------------------------------------------
    # DecisionMethod interface
    # ------------------------------------------------------------------
    def decide(self, state):
        return int(np.argmax(self.Q[self._encode(state)]))

    def memory_bytes(self):
        return self.Q.nbytes

    def train(self, env_factory):
        alpha     = 0.1
        gamma     = 0.95
        eps       = 1.0
        eps_min   = 0.05
        eps_decay = 0.995
        rng = np.random.default_rng(0)

        for _ in range(200):
            env   = env_factory()
            state = env.get_state()
            s_idx = self._encode(state)
            done  = False
            prev_fail = env.failures
            prev_tx   = env.transmissions

            while not done:
                # Epsilon-greedy action
                a = (rng.integers(0, self.N_ACTIONS)
                     if rng.random() < eps
                     else int(np.argmax(self.Q[s_idx])))

                next_state, done = env.step(a)

                # Reward shaping
                r = 0.1
                if env.transmissions > prev_tx:  r += 10
                if env.failures > prev_fail:     r -= 100
                if env.v_batt < 3.0:             r -= 5
                if state["theta"] < -20 and a == 0: r -= 3
                if state["theta"] >   0 and a == 2: r -= 2
                prev_fail, prev_tx = env.failures, env.transmissions

                s_next = self._encode(next_state)
                target = r if done else r + gamma * self.Q[s_next].max()
                self.Q[s_idx, a] += alpha * (target - self.Q[s_idx, a])
                s_idx = s_next
                state = next_state

            eps = max(eps_min, eps * eps_decay)
