import json
import tempfile
import unittest
from pathlib import Path

from llm_lab.evaluation import TrialResult, TrialStatus
from llm_lab.evaluation.storage import JsonlResultWriter, load_trial_results


def result(trial_id: str) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        experiment_id="exp_fixture",
        task_id="task.literal.000001",
        status=TrialStatus.COMPLETED,
        generation={"output_text": "answer"},
    )


class JsonlResultWriterTests(unittest.TestCase):
    def test_writer_appends_machine_readable_records_and_loads_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw" / "trials.jsonl"
            writer = JsonlResultWriter(path)
            writer.append(result("trial-1"))
            writer.append(result("trial-2"))

            records = load_trial_results(path)
            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(["trial-1", "trial-2"], [item.trial_id for item in records])
        self.assertEqual(2, len(lines))
        self.assertEqual(1, json.loads(lines[0])["schema_version"])

    def test_writer_rejects_duplicate_ids_in_memory_and_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.jsonl"
            writer = JsonlResultWriter(path)
            writer.append(result("trial-1"))

            with self.assertRaises(ValueError):
                writer.append(result("trial-1"))

            second_writer = JsonlResultWriter(path)
            with self.assertRaises(ValueError):
                second_writer.append(result("trial-1"))

    def test_loader_rejects_malformed_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.jsonl"
            path.write_text("not json\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_trial_results(path)


if __name__ == "__main__":
    unittest.main()
