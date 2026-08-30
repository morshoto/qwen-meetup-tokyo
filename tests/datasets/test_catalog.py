import json
import unittest
from pathlib import Path

from llm_lab.datasets.catalog import TaskCatalog


REPOSITORY_ROOT = Path(__file__).parents[2]


class TaskCatalogTests(unittest.TestCase):
    def test_core_catalog_contains_machine_checkable_core_task_types(self) -> None:
        catalog = TaskCatalog.from_jsonl(
            REPOSITORY_ROOT / "data" / "tasks" / "core.v001.jsonl"
        )

        self.assertEqual(
            {"literal_retrieval", "semantic_retrieval", "multi_hop"},
            {task.task_type for task in catalog.tasks},
        )
        self.assertEqual(3, len(catalog.tasks))
        self.assertEqual(
            {"exact", "normalized_exact", "exact"},
            {task.expected["type"] for task in catalog.tasks},
        )
        self.assertEqual(
            {"task.literal.000001", "task.semantic.000001", "task.multihop.000001"},
            set(catalog.ids),
        )

    def test_core_v002_has_ten_independent_presentation_tasks_per_family(self) -> None:
        catalog = TaskCatalog.from_jsonl(
            REPOSITORY_ROOT / "data" / "tasks" / "core.v002.jsonl"
        )

        self.assertEqual(30, len(catalog.tasks))
        self.assertEqual(
            {
                "literal_retrieval": 10,
                "semantic_retrieval": 10,
                "multi_hop": 10,
            },
            {
                task_type: sum(task.task_type == task_type for task in catalog.tasks)
                for task_type in {
                    "literal_retrieval",
                    "semantic_retrieval",
                    "multi_hop",
                }
            },
        )
        self.assertEqual(30, len({task.metadata["seed"] for task in catalog.tasks}))
        self.assertTrue(
            all(task.metadata["independent"] for task in catalog.tasks)
        )
        self.assertTrue(
            all(task.metadata["presentation_ready"] for task in catalog.tasks)
        )
        self.assertTrue(
            all(task.metadata["license"] == "CC0-1.0" for task in catalog.tasks)
        )
        self.assertTrue(
            all(task.expected.get("type") in {"exact", "normalized_exact"}
                for task in catalog.tasks)
        )
        self.assertTrue(
            all(
                task.expected.get("value")
                and task.evidence
                and all(item.get("text") for item in task.evidence)
                for task in catalog.tasks
            )
        )

    def test_catalog_rejects_duplicate_ids_and_missing_scorer_metadata(self) -> None:
        invalid_records = [
            {
                "schema_version": 1,
                "id": "task.duplicate",
                "type": "literal_retrieval",
                "version": 1,
                "question": "Which value?",
                "expected": {"type": "exact", "value": "one"},
                "evidence": [{"id": "e1", "text": "The value is one."}],
                "metadata": {"seed": 1, "source": "fixture", "license": "CC0-1.0"},
            },
            {
                "schema_version": 1,
                "id": "task.duplicate",
                "type": "literal_retrieval",
                "version": 1,
                "question": "Which value?",
                "expected": {"type": "exact", "value": "two"},
                "evidence": [{"id": "e1", "text": "The value is two."}],
                "metadata": {"seed": 2, "source": "fixture", "license": "CC0-1.0"},
            },
        ]

        with self.assertRaises(ValueError):
            TaskCatalog.from_records(invalid_records)

        missing_scorer = dict(invalid_records[0])
        missing_scorer["expected"] = {"value": "one"}
        with self.assertRaises(ValueError):
            TaskCatalog.from_records([missing_scorer])

    def test_fixture_manifest_points_to_versioned_prompt_and_task_catalog(self) -> None:
        fixture = json.loads(
            (
                REPOSITORY_ROOT
                / "data"
                / "fixtures"
                / "core.v001.json"
            ).read_text()
        )

        self.assertEqual("fixture.core.v001", fixture["id"])
        self.assertEqual("prompts/prompt.qa.v001.txt", fixture["prompt"])
        self.assertEqual("tasks/core.v001.jsonl", fixture["tasks"])
        self.assertEqual(1234, fixture["context"]["seed"])


if __name__ == "__main__":
    unittest.main()
