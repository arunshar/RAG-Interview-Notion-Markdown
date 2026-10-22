"""Regression tests for the export's GitHub math compatibility checks."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate_math import fenced_ranges, scan_math, validate_text


class MathValidationTests(unittest.TestCase):
    def check_valid(self, text):
        errors, _ = validate_text(text)
        self.assertEqual([], errors)

    def test_inline_operator(self):
        self.check_valid(r"Retriever: $`\mathop{\mathrm{RET}}\nolimits(q,D)`$.")

    def test_table_named_bars(self):
        self.check_valid("| Value |\n| --- |\n| $`\\lVert x\\rVert_2`$ |\n")

    def test_table_raw_bars_rejected(self):
        errors, _ = validate_text("| $`\\|x\\|`$ |")
        self.assertTrue(any("named bar" in e for e in errors))

    def test_display_boundaries(self):
        self.check_valid("Before.\n\n$$\nx_1+y_2=3\n$$\n\nAfter.\n")

    def test_missing_blank_lines_rejected(self):
        errors, _ = validate_text("Before.\n$$\nx_1=2\n$$\nAfter.\n")
        self.assertEqual(2, len(errors))

    def test_blocked_macro_rejected(self):
        errors, _ = validate_text(r"$`\operatorname{RET}(q,D)`$")
        self.assertTrue(any("GitHub rejects" in e for e in errors))

    def test_escaped_braces_and_row_breaks(self):
        self.check_valid(r"$`\begin{aligned}a&=\{1\}\\b&={2}\end{aligned}`$")

    def test_unbalanced_braces_rejected(self):
        errors, _ = validate_text(r"$`\frac{1}{2`$")
        self.assertTrue(any("Unclosed math brace" in e for e in errors))

    def test_mismatched_environment_rejected(self):
        errors, _ = validate_text(r"$`\begin{aligned}x\end{matrix}`$")
        self.assertTrue(any("environment" in e for e in errors))

    def test_fenced_literals_untouched(self):
        text = "````text\n```\n$$\n\\operatorname{example}\n```\n````\n"
        self.check_valid(text)
        self.assertEqual([(0, len(text))], fenced_ranges(text))
        self.assertEqual([], scan_math(text)[0])

    def test_inline_code_may_describe_tex(self):
        self.check_valid(r"Use `\operatorname` only in renderers that support it.")

    def test_raw_tex_rejected(self):
        errors, _ = validate_text(r"Orphan \theta in a paragraph.")
        self.assertTrue(any("raw LaTeX" in e for e in errors))

    def test_unpaired_display_rejected(self):
        errors, _ = validate_text("$$\nx\n")
        self.assertTrue(any("Unpaired display" in e for e in errors))

    def test_unpaired_inline_rejected(self):
        errors, _ = validate_text("Example $`x with a missing close.")
        self.assertTrue(any("Unpaired inline" in e for e in errors))

    def test_counts(self):
        _, stats = validate_text("Inline $`x`$.\n\n$$\ny\n$$\n")
        self.assertEqual({"inline": 1, "display": 1}, stats)


if __name__ == "__main__":
    unittest.main()