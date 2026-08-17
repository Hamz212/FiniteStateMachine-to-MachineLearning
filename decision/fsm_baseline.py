"""
FSM baseline — original power manager from Pourmoslemi et al., I2MTC 2026.

Fixed thresholds, no learning. Used as the reference for all comparisons.
"""

from decision.base import DecisionMethod


class FSMBaseline(DecisionMethod):
    name = "FSM Baseline"
    description = "Original FSM from the I2MTC 2026 paper. Fixed thresholds, no learning."
    learning_type = "rule-based"

    THETA_L  = -20.0   # °C
    V_BATT_L =  3.0    # V
    V_SC_L   =  2.2    # V

    def decide(self, state):
        if state["v_batt"] < self.V_BATT_L:
            return 2   # DEEP_CONSERVE
        if state["theta"] < self.THETA_L:
            return 1   # LOW_TEMP
        return 0       # NORMAL

    def memory_bytes(self):
        return 6       # 3 thresholds × 2 bytes
