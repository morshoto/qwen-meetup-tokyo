import time
import unittest

from llm_lab.evaluation.isolated_probe import run_isolated_probe


def slow_probe(payload):
    time.sleep(float(payload["sleep_seconds"]))
    return {"value": "completed"}


def fast_probe(payload):
    return {"value": payload["value"]}


class IsolatedProbeTests(unittest.TestCase):
    def test_timeout_terminates_child_and_returns_timeout_outcome(self) -> None:
        outcome = run_isolated_probe(
            f"{__name__}:slow_probe",
            {"sleep_seconds": 2},
            timeout_seconds=0.1,
            memory_sample_interval=0.01,
        )

        self.assertTrue(outcome.timed_out)
        self.assertIsNone(outcome.value)
        self.assertEqual("timeout", outcome.termination_reason)
        self.assertIsNotNone(outcome.exit_code)
        self.assertFalse(outcome.alive)

    def test_completed_probe_preserves_value_and_exit_code(self) -> None:
        outcome = run_isolated_probe(
            f"{__name__}:fast_probe",
            {"value": "ok"},
            timeout_seconds=2,
            memory_sample_interval=0.01,
        )

        self.assertFalse(outcome.timed_out)
        self.assertEqual({"value": "ok"}, outcome.value)
        self.assertEqual(0, outcome.exit_code)
        self.assertFalse(outcome.alive)

    def test_timeout_requires_positive_duration(self) -> None:
        with self.assertRaisesRegex(ValueError, "timeout_seconds must be positive"):
            run_isolated_probe(
                f"{__name__}:fast_probe",
                {},
                timeout_seconds=0,
            )


if __name__ == "__main__":
    unittest.main()
