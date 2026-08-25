import unittest
from pathlib import Path

from llm_lab.generation import (
    GenerationResponse,
    GenerationTiming,
    TokenUsage,
)
from llm_lab.telemetry import TelemetryCollector, capture_environment


class StepClock:
    def __init__(self) -> None:
        self.values = iter((10.0, 10.75))

    def __call__(self) -> float:
        return next(self.values)


class TelemetryCollectorTests(unittest.TestCase):
    def test_collector_derives_latency_and_token_rates_from_response_metadata(self) -> None:
        collector = TelemetryCollector(
            clock=StepClock(),
            memory_reader=lambda: (150, "fixture"),
            environment_factory=lambda: {"platform": "fixture"},
        )
        handle = collector.start()
        record = collector.finish(
            handle,
            GenerationResponse(
                output_text="answer",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
                timing=GenerationTiming(
                    ttft_seconds=0.1,
                    prefill_seconds=0.5,
                    decode_seconds=0.25,
                    total_seconds=0.75,
                ),
            ),
        )

        self.assertEqual(0.75, record.total_seconds)
        self.assertEqual(0.1, record.ttft_seconds)
        self.assertEqual(20.0, record.prefill_tokens_per_second)
        self.assertEqual(20.0, record.decode_tokens_per_second)
        self.assertEqual(150, record.peak_memory_bytes)
        self.assertEqual("fixture", record.memory_measurement)
        self.assertEqual("fixture", record.environment["platform"])
        self.assertEqual(20.0, record.to_dict()["prefill_tokens_per_second"])

    def test_environment_capture_is_machine_readable_and_git_sha_is_optional(self) -> None:
        environment = capture_environment(Path.cwd())

        self.assertIn("python_version", environment)
        self.assertIn("platform", environment)
        self.assertIn("machine", environment)
        self.assertIn("git_sha", environment)

    def test_missing_response_still_records_wall_clock_and_memory(self) -> None:
        collector = TelemetryCollector(
            clock=StepClock(),
            memory_reader=lambda: (200, "fixture"),
            environment_factory=lambda: {},
        )

        record = collector.finish(collector.start(), None)

        self.assertEqual(0.75, record.total_seconds)
        self.assertIsNone(record.ttft_seconds)
        self.assertEqual(200, record.peak_memory_bytes)


if __name__ == "__main__":
    unittest.main()
