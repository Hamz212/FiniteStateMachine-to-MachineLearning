"""
FSM with thresholds optimised by a genetic algorithm.

Same three-condition structure as the baseline FSM, but the threshold
values (theta_L, v_batt_L) are tuned by a simple steady-state GA to
maximise a weighted combination of autonomy, transmissions, and failures.
"""

import numpy as np
from decision.base import DecisionMethod


class FSMOptimized(DecisionMethod):
    name = "FSM Optimisée (GA)"
    description = "FSM whose thresholds are tuned by a genetic algorithm."
    learning_type = "optimization"

    def __init__(self):
        self.theta_l  = -20.0
        self.v_batt_l =  3.0
        self.v_sc_l   =  2.2

    def decide(self, state):
        if state["v_batt"] < self.v_batt_l:
            return 2
        if state["theta"] < self.theta_l:
            return 1
        return 0

    def memory_bytes(self):
        return 6   # 3 thresholds × 2 bytes

    def train(self, env_factory):
        rng = np.random.default_rng(42)
        POP_SIZE     = 20
        N_GEN        = 15
        MUTATION_STD = 0.3

        population = [
            (rng.uniform(-30, -10), rng.uniform(2.9, 3.3), rng.uniform(2.0, 2.6))
            for _ in range(POP_SIZE)
        ]

        def fitness(ind):
            self.theta_l, self.v_batt_l, self.v_sc_l = ind
            env = env_factory()
            done = False
            while not done:
                _, done = env.step(self.decide(env.get_state()))
            return (env.autonomy_days() * 100
                    + env.transmissions * 0.5
                    - env.failures * 1000)

        for _ in range(N_GEN):
            scored  = sorted([(fitness(ind), ind) for ind in population],
                             reverse=True, key=lambda x: x[0])
            elites  = [ind for _, ind in scored[:POP_SIZE // 4]]
            new_pop = list(elites)
            while len(new_pop) < POP_SIZE:
                i1, i2 = rng.choice(len(elites), 2, replace=True)
                child = []
                for a, b in zip(elites[i1], elites[i2]):
                    v = (a + b) / 2
                    if rng.random() < MUTATION_STD:
                        v += rng.normal(0, 0.3)
                    child.append(v)
                new_pop.append(tuple(child))
            population = new_pop

        _, best = max((fitness(ind), ind) for ind in population)
        self.theta_l, self.v_batt_l, self.v_sc_l = best
