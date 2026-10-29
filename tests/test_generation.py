"""Regression test for decoder chat-template inputs."""
import unittest

import torch

from src.generate import generate_one


class _Inputs(dict):
    def to(self, device):
        return self


class _Tokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["return_dict"] is True
        assert kwargs["add_generation_prompt"] is True
        return _Inputs(input_ids=torch.tensor([[1, 4]]),
                       attention_mask=torch.tensor([[1, 1]]))

    def decode(self, tokens, skip_special_tokens=True):
        assert tokens.tolist() == [5, 2]
        return "हिंदी"


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.generation_config = type("GenerationConfig", (), {"eos_token_id": 2})()
        self.saw_attention_mask = False

    def generate(self, **kwargs):
        self.saw_attention_mask = torch.equal(kwargs["attention_mask"],
                                              torch.tensor([[1, 1]]))
        return torch.tensor([[1, 4, 5, 2]])


class DecoderGenerationTests(unittest.TestCase):
    def test_chat_generation_passes_attention_mask_and_slices_continuation(self):
        model = _Model()
        output, truncated = generate_one("llama", model, _Tokenizer(), "hello",
                                         {"max_new_tokens": 128}, None)
        self.assertTrue(model.saw_attention_mask)
        self.assertEqual(output, "हिंदी")
        self.assertFalse(truncated)


if __name__ == "__main__":
    unittest.main()
