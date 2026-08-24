import unittest

from llm_lab.context.synthetic import Evidence, SyntheticContextGenerator


class SyntheticContextGeneratorTests(unittest.TestCase):
    def test_generation_is_reproducible_at_an_exact_target_token_count(self) -> None:
        generator = SyntheticContextGenerator()
        evidence = Evidence(id="evidence-1", text="The answer is ZX-4817.")

        first = generator.generate(
            [evidence], target_tokens=48, evidence_position=0.25, seed=1234
        )
        second = generator.generate(
            [evidence], target_tokens=48, evidence_position=0.25, seed=1234
        )

        self.assertEqual(first, second)
        self.assertEqual(48, first.token_count)
        self.assertIn(evidence.text, first.text)
        self.assertEqual(1, len(first.evidence))
        self.assertEqual(0.25, first.evidence[0].requested_position)
        self.assertAlmostEqual(0.25, first.evidence[0].actual_position, delta=0.05)

    def test_evidence_position_changes_without_changing_seeded_filler(self) -> None:
        generator = SyntheticContextGenerator()
        evidence = Evidence(id="evidence-1", text="The answer is ZX-4817.")

        early = generator.generate(
            [evidence], target_tokens=48, evidence_position=0.1, seed=1234
        )
        late = generator.generate(
            [evidence], target_tokens=48, evidence_position=0.9, seed=1234
        )

        self.assertLess(early.evidence[0].token_start, late.evidence[0].token_start)
        self.assertEqual(48, late.token_count)
        self.assertEqual(
            early.text.split()[:3],
            late.text.split()[:3],
        )

    def test_generator_rejects_invalid_positions_and_short_targets(self) -> None:
        generator = SyntheticContextGenerator()
        evidence = Evidence(id="evidence-1", text="one two three")

        with self.assertRaises(ValueError):
            generator.generate([evidence], target_tokens=10, evidence_position=1.1, seed=1)
        with self.assertRaises(ValueError):
            generator.generate([evidence], target_tokens=2, evidence_position=0.5, seed=1)


if __name__ == "__main__":
    unittest.main()
