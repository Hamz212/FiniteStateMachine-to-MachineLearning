"""
IoT Node Power Manager Environment — Benchmark version
=======================================================
Calibrated from Pourmoslemi et al., I2MTC 2026.
3 actions: NORMAL, LOW_TEMP, DEEP_CONSERVE.
"""

import numpy as np


class IoTNodeEnv:
    # Physical parameters (paper)
    V_BATT_NOMINAL = 3.7
    V_BATT_MIN = 2.8
    BATT_CAPACITY_J = 140e-3 * 3.7 * 3600       # ~1865 J
    SC_CAPACITANCE = 0.22
    SC_V_MAX = 3.3
    SC_V_MIN = 2.1
    I_BASELINE = 4.16e-6
    E_RF_TX = 18e-3
    E_MEASUREMENT = 4.16e-6 * 3.7 * 60

    # Action space
    NORMAL, LOW_TEMP, DEEP_CONSERVE = 0, 1, 2
    N_ACTIONS = 3
    ACTION_NAMES = ["NORMAL", "LOW_TEMP", "DEEP_CONSERVE"]

    @staticmethod
    def batt_capacity_factor(theta):
        if theta < -30: return 0.30
        elif theta < -20: return 0.55
        elif theta < 0: return 0.85
        elif theta < 60: return 1.00
        else: return 0.90

    @staticmethod
    def sc_leakage_rate(theta):
        if theta < -20: return 0.0005
        elif theta < 25: return 0.002
        else: return 0.005

    def __init__(self, seed=0, episode_minutes=60*24*7, profile=None):
        self.rng = np.random.default_rng(seed)
        self.episode_minutes = episode_minutes
        self.preset_profile = profile
        self.reset()

    def reset(self):
        self.t = 0
        self.theta = 20.0
        self.v_batt = 3.7
        self.v_sc = 3.0
        self.packets_pending = 0
        self.batt_energy_remaining = self.BATT_CAPACITY_J
        self.failures = 0
        self.transmissions = 0
        self.dropped_packets = 0
        self.total_energy_consumed = 0.0
        self.last_tx_time = 0
        self.tx_intervals = []

        if self.preset_profile is not None:
            self.temperature_profile = self.preset_profile.copy()
        else:
            self._gen_temperature_profile()
        return self.get_state()

    def _gen_temperature_profile(self):
        """Standard TVCC-like cycle."""
        n = self.episode_minutes
        t = np.arange(n)
        cycle1 = 50 * np.sin(2*np.pi*t/(60*8))      # 8h cycle, ±50°C
        cycle2 = 20 * np.sin(2*np.pi*t/(60*1.5))    # 1.5h cycle, ±20°C
        noise = self.rng.normal(0, 2, n)
        self.temperature_profile = np.clip(cycle1+cycle2+noise, -40, 85)

    def get_state(self):
        """Continuous state dict — used by all decision methods."""
        return {
            "theta": float(self.theta),
            "v_batt": float(self.v_batt),
            "v_sc": float(self.v_sc),
            "packets_pending": int(self.packets_pending),
        }

    def _battery_voltage_from_energy(self):
        soc = max(0, self.batt_energy_remaining / self.BATT_CAPACITY_J)
        return self.V_BATT_MIN + (self.V_BATT_NOMINAL - self.V_BATT_MIN) * soc

    def step(self, action):
        done = False
        energy_used_this_step = 0.0

        # Temperature update
        self.theta = float(self.temperature_profile[self.t])
        # SC leakage
        self.v_sc = max(self.SC_V_MIN - 0.5,
                        self.v_sc - self.sc_leakage_rate(self.theta))

        if action == self.NORMAL:
            self.packets_pending += 1
            e = self.E_MEASUREMENT
            self.batt_energy_remaining -= e
            energy_used_this_step += e
            if self.v_sc < 2.9:
                charge = min(0.4, self.SC_V_MAX - self.v_sc)
                ec = 0.5 * self.SC_CAPACITANCE * (
                    (self.v_sc+charge)**2 - self.v_sc**2)
                ec_eff = ec / max(self.batt_capacity_factor(self.theta), 0.1)
                self.batt_energy_remaining -= ec_eff
                energy_used_this_step += ec_eff
                self.v_sc += charge
            if self.packets_pending >= 4 and self.v_sc > 2.5:
                self.v_sc -= 0.3
                self.packets_pending = 0
                self.transmissions += 1
                self.tx_intervals.append(self.t - self.last_tx_time)
                self.last_tx_time = self.t

        elif action == self.LOW_TEMP:
            self.packets_pending += 1
            e = self.E_MEASUREMENT * 0.5
            self.batt_energy_remaining -= e
            energy_used_this_step += e
            if self.v_sc < 3.2:
                charge = min(0.6, self.SC_V_MAX - self.v_sc)
                ec = 0.5 * self.SC_CAPACITANCE * (
                    (self.v_sc+charge)**2 - self.v_sc**2)
                ec_eff = ec / max(self.batt_capacity_factor(self.theta), 0.1)
                self.batt_energy_remaining -= ec_eff
                energy_used_this_step += ec_eff
                self.v_sc += charge
            if self.packets_pending > 16:
                self.dropped_packets += self.packets_pending - 16
                self.packets_pending = 16

        elif action == self.DEEP_CONSERVE:
            self.packets_pending += 1
            e = self.E_MEASUREMENT * 0.3
            self.batt_energy_remaining -= e
            energy_used_this_step += e
            if self.packets_pending > 16:
                self.dropped_packets += self.packets_pending - 16
                self.packets_pending = 16

        self.total_energy_consumed += energy_used_this_step
        self.v_batt = self._battery_voltage_from_energy()
        self.v_sc = float(np.clip(self.v_sc, 1.5, self.SC_V_MAX))

        if self.v_batt < self.V_BATT_MIN or self.batt_energy_remaining <= 0:
            self.failures += 1
            done = True

        self.t += 1
        if self.t >= self.episode_minutes:
            done = True

        return self.get_state(), done

    def autonomy_days(self):
        """Survived time in days."""
        return self.t / (60 * 24)
