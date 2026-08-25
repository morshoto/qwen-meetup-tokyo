import unittest

from llm_lab.context import Evidence, TokenizerContextGenerator


class FixtureTokenizer:
    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids: list[int] = []
        for token in text.split():
            if token not in self._token_to_id:
                token_id = len(self._token_to_id) + 1
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token
            ids.append(self._token_to_id[token])
        return ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return " ".join(self._id_to_token[token_id] for token_id in token_ids)


class TokenizerContextGeneratorTests(unittest.TestCase):
    def test_generation_uses_tokenizer_tokens_for_length_and_positions(self) -> None:
        generator = TokenizerContextGenerator(FixtureTokenizer())
        evidence = [Evidence(id="answer", text="The answer is ZX-4817.")]

        generated = generator.generate(
            evidence,
            target_tokens=32,
            evidence_position=0.25,
            seed=1234,
        )

        self.assertEqual(32, generated.token_count)
        self.assertEqual("tokenizer-v1", generated.metadata["tokenization"])
        self.assertEqual("The answer is ZX-4817.", evidence[0].text)
        self.assertLess(generated.evidence[0].token_start, generated.evidence[0].token_end)
        self.assertAlmostEqual(0.25, generated.evidence[0].actual_position, delta=0.05)

    def test_generation_is_reproducible_with_same_tokenizer_and_seed(self) -> None:
        first = TokenizerContextGenerator(FixtureTokenizer()).generate(
            [Evidence(id="answer", text="The answer is ZX-4817.")],
            target_tokens=32,
            evidence_position=0.75,
            seed=1234,
        )
        second = TokenizerContextGenerator(FixtureTokenizer()).generate(
            [Evidence(id="answer", text="The answer is ZX-4817.")],
            target_tokens=32,
            evidence_position=0.75,
            seed=1234,
        )

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
