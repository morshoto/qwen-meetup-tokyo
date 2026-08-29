import tempfile
import unittest
from pathlib import Path

from llm_lab.analysis.agent_reliability import (
    AgentAnalysisError,
    aggregate_agent_trials,
    require_measured_trials,
)
from llm_lab.evaluation import TrialResult, TrialStatus


def trial(
    trial_id: str,
    *,
    variant: str,
    length: int,
    position: float,
    correct: bool | None,
    category: str,
    fixture_only: bool = False,
    tokens: int = 100,
    tool_calls: int = 1,
    valid_tool_calls: int = 1,
    repeated: int = 0,
    recovery: int = 0,
) -> TrialResult:
    metrics = {
        "trajectory_length": length,
        "requested_critical_position": position,
        "critical_fact_reused": correct is True,
        "tool_call_n": tool_calls,
        "valid_tool_call_n": valid_tool_calls,
        "repeated_action_n": repeated,
        "recovery_n": recovery,
        "total_input_tokens": tokens,
        "failure_category": category,
    }
    return TrialResult(
        trial_id=trial_id,
        experiment_id="exp_004",
        task_id=trial_id,
        status=TrialStatus.COMPLETED if correct is not None else TrialStatus.RUNTIME_ERROR,
        input={
            "task_type": "agent_state_tracking",
            "variant_condition_id": variant,
            "trajectory_length": length,
            "requested_critical_position": position,
            "fixture_only": fixture_only,
            "metrics": metrics,
        },
        score={"correct": correct} if correct is not None else {},
        environment={
            "purpose": "harness_smoke_only" if fixture_only else "measurement"
        },
    )


class AgentReliabilityAnalysisTests(unittest.TestCase):
    def test_analysis_rows_preserve_length_position_and_weighted_outcomes(self) -> None:
        rows = aggregate_agent_trials(
            [
                trial(
                    "one",
                    variant="q8_0",
                    length=8,
                    position=0.5,
                    correct=True,
                    category="success",
                    tokens=120,
                ),
                trial(
                    "two",
                    variant="q8_0",
                    length=8,
                    position=0.5,
                    correct=False,
                    category="state_tracking",
                    tokens=160,
                    repeated=1,
                    recovery=1,
                ),
                trial(
                    "three",
                    variant="q8_0",
                    length=8,
                    position=0.5,
                    correct=None,
                    category="runtime",
                    tokens=0,
                ),
            ]
        )

        self.assertEqual(1, len(rows))
        row = rows[0]
        self.assertEqual("q8_0", row["variant_condition_id"])
        self.assertEqual(8, row["trajectory_length"])
        self.assertEqual(0.5, row["requested_critical_position"])
        self.assertEqual(3, row["attempted_n"])
        self.assertEqual(2, row["scored_n"])
        self.assertEqual(0.5, row["scored_accuracy"])
        self.assertAlmostEqual(1 / 3, row["final_task_success"])
        self.assertEqual(280, row["total_input_tokens"])
        self.assertEqual(1, row["repeated_action_n"])
        self.assertEqual(1, row["recovery_n"])
        self.assertEqual(1, row["failure_category_counts"]["state_tracking"])
        self.assertEqual(1, row["failure_category_counts"]["runtime"])

    def test_failure_taxonomy_distinguishes_retrieval_and_planning_failures(self) -> None:
        rows = aggregate_agent_trials(
            [
                trial(
                    "retrieval",
                    variant="q4_k_m",
                    length=16,
                    position=0.05,
                    correct=False,
                    category="retrieval",
                ),
                trial(
                    "planning",
                    variant="q4_k_m",
                    length=16,
                    position=0.05,
                    correct=False,
                    category="tool_planning",
                    tool_calls=2,
                    valid_tool_calls=1,
                ),
            ]
        )

        self.assertEqual(
            {"retrieval": 1, "tool_planning": 1},
            rows[0]["failure_category_counts"],
        )
        self.assertAlmostEqual(2 / 3, rows[0]["tool_call_validity"])

    def test_analysis_rejects_missing_dimensions_and_fixture_only_trials(self) -> None:
        missing_dimension = trial(
            "missing",
            variant="q8_0",
            length=4,
            position=0.5,
            correct=True,
            category="success",
        )
        missing_dimension.input.pop("requested_critical_position")
        with self.assertRaises(AgentAnalysisError):
            aggregate_agent_trials([missing_dimension])

        fixture = trial(
            "fixture",
            variant="q8_0",
            length=4,
            position=0.5,
            correct=True,
            category="success",
            fixture_only=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(AgentAnalysisError):
                require_measured_trials([fixture], output_directory=Path(directory))


if __name__ == "__main__":
    unittest.main()
