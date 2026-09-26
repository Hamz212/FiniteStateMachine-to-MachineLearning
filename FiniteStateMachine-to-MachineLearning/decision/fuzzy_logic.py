"""
Fuzzy logic controller for the IoT node power manager.

Trapezoidal membership functions soften the hard thresholds of the FSM,
reducing chattering near decision boundaries. Defuzzification uses a
weighted sum of rule activations.
"""

import numpy as np
from decision.base import DecisionMethod


def _trapezoid(x, a, b, c, d):
    """Trapezoidal membership function, returns value in [0, 1]."""
    if x <= a or x >= d:
        return 0.0
    if x < b:
        return (x - a) / (b - a)
    if x <= c:
        return 1.0
    return (d - x) / (d - c)


class FuzzyLogic(DecisionMethod):
    name = "Logique Floue"
    description = "Fuzzy controller with smooth membership functions and weighted defuzzification."
    learning_type = "rule-based"

    def decide(self, state):
        theta  = state["theta"]
        v_batt = state["v_batt"]
        v_sc   = state["v_sc"]

        # Temperature membership
        cold     = _trapezoid(theta,  -50, -40, -22, -15)
        moderate = _trapezoid(theta,  -22, -15,  30,  45)
        warm     = _trapezoid(theta,   30,  45,  85, 100)

        # Battery membership
        low_batt = _trapezoid(v_batt, 2.6, 2.8, 3.0, 3.15)
        ok_batt  = _trapezoid(v_batt, 3.0, 3.15, 3.7, 3.9)

        # SC membership
        low_sc = _trapezoid(v_sc, 1.8, 2.0, 2.2, 2.5)
        ok_sc  = _trapezoid(v_sc, 2.2, 2.5, 3.3, 3.5)

        scores = [0.0, 0.0, 0.0]
        scores[1] += cold
        scores[2] += low_batt
        scores[0] += min(moderate, ok_batt)
        scores[0] += min(warm, ok_batt)
        scores[1] += min(low_sc, cold)
        scores[2] += min(low_batt, low_sc)

        return int(np.argmax(scores))

    def memory_bytes(self):
        return 48   # membership function parameters
