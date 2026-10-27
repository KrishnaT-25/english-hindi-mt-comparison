import tempfile
import unittest
from pathlib import Path

import torch

from src.common import ROOT
from src.models.transformer import TranslationTransformer


class TransformerTests(unittest.TestCase):
    def test_forward_backward_and_reload(self):
        model = TranslationTransformer(vocab_size=32, d_model=16, nhead=2,
                                       encoder_layers=1, decoder_layers=1,
                                       ff_dim=32, dropout=0)
        source = torch.tensor([[2, 8, 9, 3, 0]])
        target = torch.tensor([[2, 10, 11]])
        logits = model(source, target)
        self.assertEqual(tuple(logits.shape), (1, 3, 32))
        logits.square().mean().backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))
        model.eval()
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as folder:
            path = Path(folder) / "state.pt"
            torch.save(model.state_dict(), path)
            restored = TranslationTransformer(vocab_size=32, d_model=16, nhead=2,
                                              encoder_layers=1, decoder_layers=1,
                                              ff_dim=32, dropout=0).eval()
            restored.load_state_dict(torch.load(path, weights_only=True))
            with torch.no_grad():
                self.assertTrue(torch.allclose(model(source, target), restored(source, target)))


if __name__ == "__main__":
    unittest.main()

