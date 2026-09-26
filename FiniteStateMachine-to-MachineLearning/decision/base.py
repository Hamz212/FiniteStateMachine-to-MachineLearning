"""
Common interface for all power management decision methods.

Every method must implement:
  decide(state) -> int   called at each wake-up cycle
  train(env_factory)     optional offline training phase (default: no-op)
  memory_bytes() -> int  RAM footprint when deployed on the node
"""


class DecisionMethod:
    name = "unnamed"
    description = ""
    learning_type = "unknown"

    def train(self, env_factory):
        """Offline training. Receives a callable returning a fresh environment."""
        pass

    def decide(self, state):
        """
        Choose a power mode for the current cycle.

        Parameters
        ----------
        state : dict
            Keys: theta (°C), v_batt (V), v_sc (V), packets_pending (int)

        Returns
        -------
        int
            0 = NORMAL, 1 = LOW_TEMP, 2 = DEEP_CONSERVE
        """
        raise NotImplementedError

    def memory_bytes(self):
        """RAM footprint in bytes for deployment on the target node."""
        return 0
