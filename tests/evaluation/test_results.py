import json
import unittest

from llm_lab.evaluation import TrialResult, TrialStatus, make_trial_id


class TrialResultTests(unittest.TestCase):
    def test_trial_id_is_deterministic_and_distinguishes_repeats(self) -> None:
        first = make_trial_id(
            "exp_001",
            "task.literal.000001",
            condition_id="q8:ctx64:p050",
            repeat_index=1,
        )
        same = make_trial_id(
            "exp_001",
            "task.literal.000001",
            condition_id="q8:ctx64:p050",
            repeat_index=1,
        )
        second = make_trial_id(
            "exp_001",
            "task.literal.000001",
            condition_id="q8:ctx64:p050",
            repeat_index=2,
        )

        self.assertEqual("exp_001:task.literal.000001:q8:ctx64:p050:run01", first)
        self.assertEqual(first, same)
        self.assertNotEqual(first, second)

    def test_completed_and_failure_records_round_trip_as_json(self) -> None:
        completed = TrialResult(
            trial_id="exp_001:task.literal.000001:default:run01",
            experiment_id="exp_001",
            task_id="task.literal.000001",
            status=TrialStatus.COMPLETED,
            model={"id": "fixture/model", "revision": "model-sha"},
            runtime={"name": "fixture", "version": "1.0"},
            input={"task_type": "literal_retrieval", "prompt_id": "prompt.qa.v001"},
            generation={"output_text": "ZX-4817", "output_tokens": 1},
            score={"correct": True, "value": 1.0, "scorer": "expected.v1"},
            timing={"total_s": 0.25},
            memory={"peak_bytes": 100},
            environment={"python": "3.13"},
        )
        runtime_error = TrialResult(
            trial_id="exp_001:task.literal.000001:default:run02",
            experiment_id="exp_001",
            task_id="task.literal.000001",
            status=TrialStatus.RUNTIME_ERROR,
            error={"type": "RuntimeError", "message": "backend unavailable"},
        )

        record = completed.to_record()
        encoded = json.dumps(record, sort_keys=True)
        decoded = TrialResult.from_record(json.loads(encoded))

        self.assertEqual(1, record["schema_version"])
        self.assertEqual(TrialStatus.COMPLETED, decoded.status)
        self.assertEqual("ZX-4817", decoded.generation["output_text"])
        self.assertEqual("backend unavailable", runtime_error.to_record()["error"]["message"])

    def test_result_rejects_unknown_status_or_schema_version(self) -> None:
        with self.assertRaises(ValueError):
            TrialResult(
                trial_id="trial",
                experiment_id="exp",
                task_id="task",
                status="unknown",
            )

        with self.assertRaises(ValueError):
            TrialResult.from_record(
                {
                    "schema_version": 99,
                    "trial_id": "trial",
                    "experiment_id": "exp",
                    "task_id": "task",
                    "status": "completed",
                }
            )


if __name__ == "__main__":
    unittest.main()
