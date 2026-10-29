import tempfile
import unittest
from pathlib import Path

from src.common import ROOT, read_jsonl, sha256, write_jsonl
from src.evaluate import expected_ids, score
from src.prepare_data import collect, normalize


class PreparationTests(unittest.TestCase):
    def test_normalize_and_no_overlap(self):
        split = [
            {"translation": {"en": " A ", "hi": " हिंदी "}},
            {"translation": {"en": "A", "hi": "हिंदी"}},
            {"translation": {"en": "B", "hi": "अन्य"}},
            {"translation": {"en": "", "hi": "रिक्त"}},
        ]
        rows, stats = collect(split, "train", {"B"})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "A")
        self.assertEqual(stats["removed"], {"empty": 1, "duplicate_pair": 1, "source_overlap": 1})
        self.assertEqual(normalize("e\u0301"), "é")

    def test_deterministic_file(self):
        rows = [{"id": "x", "source": "Hello", "reference": "नमस्ते"}]
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as folder:
            path = Path(folder) / "rows.jsonl"
            write_jsonl(path, rows)
            first = sha256(path)
            write_jsonl(path, rows)
            self.assertEqual(sha256(path), first)
            self.assertEqual(list(read_jsonl(path)), rows)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.refs = [{"id": "a", "reference": "यह एक किताब है"},
                     {"id": "b", "reference": "नमस्ते दुनिया"}]
        self.perfect = [{"id": x["id"], "reference": x["reference"],
                         "prediction": x["reference"]} for x in self.refs]

    def test_perfect_and_order(self):
        first = score(self.perfect, self.refs)
        second = score(list(reversed(self.perfect)), self.refs)
        self.assertAlmostEqual(first["BLEU"], second["BLEU"])
        self.assertAlmostEqual(first["chrF"], 100.0)
        self.assertAlmostEqual(first["chrF"], second["chrF"])

    def test_missing_and_duplicate(self):
        with self.assertRaises(ValueError):
            score(self.perfect[:1], self.refs)
        with self.assertRaises(ValueError):
            score(self.perfect + self.perfect[:1], self.refs)

    def test_empty_scores_worse(self):
        empty = [{**x, "prediction": ""} for x in self.perfect]
        self.assertLess(score(empty, self.refs)["chrF"], score(self.perfect, self.refs)["chrF"])

    def test_explicit_test_subset_must_match_frozen_prefix(self):
        manifest = {"ids": ["a", "b", "c", "d"]}
        self.assertEqual(expected_ids(manifest, "test", 2, 2), ["a", "b"])
        with self.assertRaises(ValueError):
            expected_ids(manifest, "test", 2, 1)
        with self.assertRaises(ValueError):
            expected_ids(manifest, "validation", 2, 2)


if __name__ == "__main__":
    unittest.main()

