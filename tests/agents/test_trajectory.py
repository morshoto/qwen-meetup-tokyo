import json
import unittest

from llm_lab.agents.trajectory import (
    ActionParseError,
    AgentTrajectory,
    DeterministicToolEnvironment,
    parse_action,
)


class AgentTrajectoryTests(unittest.TestCase):
    def test_trajectory_round_trips_ordered_tool_events(self) -> None:
        trajectory = AgentTrajectory()
        trajectory.add_message("user", "Find the target module.")
        trajectory.add_message(
            "assistant",
            '{"action":"tool","name":"discover_fact","arguments":{}}',
            source="model",
            metadata={"valid": True},
        )
        trajectory.add_tool_result(
            "discover_fact",
            "middleware/auth.ts",
            metadata={"critical_observation": True},
        )

        encoded = json.dumps(trajectory.to_records(), sort_keys=True)
        restored = AgentTrajectory.from_records(json.loads(encoded))

        self.assertEqual(trajectory.to_records(), restored.to_records())
        self.assertEqual([0, 1, 2], [event["index"] for event in restored.to_records()])
        self.assertEqual("tool", restored.to_records()[2]["role"])
        self.assertEqual("discover_fact", restored.to_records()[2]["tool_name"])
        self.assertEqual(trajectory.fingerprint, restored.fingerprint)

    def test_action_parser_accepts_tool_and_answer_actions(self) -> None:
        tool = parse_action(
            '{"action":"tool","name":"inspect_noise","arguments":{"index":2}}'
        )
        answer = parse_action('{"action":"answer","value":"middleware/auth.ts"}')

        self.assertEqual("tool", tool.action)
        self.assertEqual("inspect_noise", tool.name)
        self.assertEqual({"index": 2}, tool.arguments)
        self.assertEqual("answer", answer.action)
        self.assertEqual("middleware/auth.ts", answer.value)

    def test_action_parser_rejects_non_object_and_unknown_actions(self) -> None:
        with self.assertRaises(ActionParseError):
            parse_action("not json")
        with self.assertRaises(ActionParseError):
            parse_action('{"action":"delete_everything"}')
        with self.assertRaises(ActionParseError):
            parse_action('{"action":"tool","name":"inspect_noise","arguments":[]}')

    def test_deterministic_environment_returns_stable_observations_and_errors(self) -> None:
        first = DeterministicToolEnvironment(
            critical_observation="middleware/auth.ts",
            distractor_outputs=("README explains deployment.", "Tests use pytest."),
        )
        second = DeterministicToolEnvironment(
            critical_observation="middleware/auth.ts",
            distractor_outputs=("README explains deployment.", "Tests use pytest."),
        )

        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(
            first.execute("discover_fact").to_record(),
            second.execute("discover_fact").to_record(),
        )
        self.assertEqual(
            "Tests use pytest.",
            first.execute("inspect_noise", {"index": 1}).output,
        )
        unknown = first.execute("missing_tool")
        self.assertFalse(unknown.ok)
        self.assertEqual("missing_tool", unknown.tool_name)
        self.assertIn("unknown tool", unknown.error or "")


if __name__ == "__main__":
    unittest.main()
