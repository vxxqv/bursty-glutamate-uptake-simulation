import unittest
from dataclasses import replace

import numpy as np

import glutamate_uptake as model


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.protocol = replace(
            model.Protocol(), dt_ms=0.04, duration_ms=520.0,
            n_events=10, first_event_ms=50.0, last_event_ms=410.0,
            burst_size=5, burst_isi_ms=20.0,
        )
        self.transporter = model.TransporterParameters()
        self.linear_rate, self.vmax, _ = model.calibrate_models(
            self.protocol, self.transporter
        )

    def run_pattern(self, clearance_model, pattern):
        times = model.event_times(pattern, self.protocol)
        amplitudes = np.full((1, self.protocol.n_events),
                             self.protocol.pulse_size_km)
        impulses, last = model.make_impulses([times], amplitudes, self.protocol)
        return model.simulate_batch(
            clearance_model, impulses, last, self.protocol,
            linear_rate=self.linear_rate, vmax=self.vmax,
            transporter=self.transporter, return_trace=True,
        )

    def test_equal_count_protocols_have_equal_release(self):
        tonic = model.event_times("tonic", self.protocol)
        burst = model.event_times("burst", self.protocol)
        self.assertEqual(len(tonic), len(burst))
        self.assertEqual(tonic[0], burst[0])
        self.assertEqual(tonic[-1], burst[-1])

    def test_linear_auc_is_timing_invariant(self):
        tonic = self.run_pattern("Linear", "tonic")["auc_km_ms"][0]
        burst = self.run_pattern("Linear", "burst")["auc_km_ms"][0]
        self.assertAlmostEqual(burst / tonic, 1.0, places=7)

    def test_transporter_state_is_bounded(self):
        result = self.run_pattern("Transporter-state", "burst")
        self.assertTrue(np.all(result["trace"] >= 0.0))
        self.assertTrue(np.all(result["free_trace"] >= 0.0))
        self.assertTrue(np.all(result["free_trace"] <= 1.0))

    def test_single_pulse_calibration_is_near_one_millisecond(self):
        t90 = model.single_pulse_t90(
            "Transporter-state", self.protocol, transporter=self.transporter
        )
        self.assertGreaterEqual(t90, 0.75)
        self.assertLessEqual(t90, 1.05)


if __name__ == "__main__":
    unittest.main()
