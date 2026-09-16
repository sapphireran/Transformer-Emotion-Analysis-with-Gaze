import sys
import unittest
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import importlib


class ExampleBuildTests(unittest.TestCase):
    def _build(self, name: str) -> str:
        module = importlib.import_module(name)
        return module.build()

    def test_inventory_mentions_subject_three(self):
        text = self._build("01_repo_inventory")
        self.assertIn("3_SR.csv", text)
        self.assertIn("299", text)

    def test_alignment_mentions_remap(self):
        text = self._build("02_subject_alignment")
        self.assertIn("246", text)
        self.assertIn("250", text)

    def test_bug_demo_mentions_last_batch(self):
        text = self._build("07_test_loop_bug")
        self.assertIn("162", text)

    def test_fusion_has_architecture(self):
        text = self._build("08_cpu_fusion")
        self.assertIn("Linear(5→16)", text)


if __name__ == "__main__":
    unittest.main()
