import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from compare_image_latency import summarize, validate


def record(us=10, checksum="0123456789abcdef"):
    return {"case": "u8/test/128x128/128x128", "loops": 16,
            "samples_us": [us * .9, us, us * 1.1], "median_us": us, "hash": checksum}


class Measurements(unittest.TestCase):
    def test_valid_record(self):
        self.assertEqual(validate(record(), record()["case"], 3), record())

    def test_invalid_records(self):
        for field, value in [("case", "wrong"), ("loops", 0), ("hash", "invalid"),
                             ("median_us", 11), ("samples_us", [10]),
                             ("samples_us", [1, float("nan"), 3]),
                             ("samples_us", [1, float("inf"), 3]),
                             ("samples_us", [1, -2, 3])]:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                item = record()
                item[field] = value
                validate(item, record()["case"], 3)

    def test_regression_limit(self):
        self.assertTrue(summarize("x", [(record(), record(10.5))])["pass"])
        self.assertFalse(summarize("x", [(record(), record(10.51))])["pass"])

    def test_control_rejects_apparent_gain(self):
        pairs = [(record(), record(9))]
        self.assertTrue(summarize("x", pairs)["pass"])
        self.assertEqual(summarize("x", pairs, control=True)["perf_status"], "UNSTABLE")

    def test_paired_median_not_ratio_of_medians(self):
        pairs = [(record(10), record(20)), (record(100), record(101)), (record(30), record(31))]
        self.assertAlmostEqual(summarize("x", pairs)["ratio"], 31 / 30)

    def test_changed_output_blocks_performance_pass(self):
        pairs = [(record(), record(9, "1123456789abcdef"))]
        self.assertFalse(summarize("x", pairs)["pass"])
        self.assertTrue(summarize("x", pairs, expected_change=True)["pass"])
        self.assertFalse(summarize("x", pairs, expected_change=True, control=True)["pass"])

    def test_expected_change_cannot_hide_nondeterminism(self):
        pairs = [(record(), record(9)), (record(), record(9, "1123456789abcdef"))]
        self.assertFalse(summarize("x", pairs, expected_change=True)["pass"])


class CommandLine(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.binary = self.root / "probe"
        self.binary.write_text(f"#!{sys.executable}\nimport json, sys\n"
                               "if '--list' in sys.argv: print('u8/test/128x128/128x128')\n"
                               f"else: print(json.dumps({record()!r}))\n")
        self.binary.chmod(0o755)
        self.script = Path(__file__).with_name("compare_image_latency.py")

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, output, *arguments):
        return subprocess.run([sys.executable, str(self.script), "--baseline", str(self.binary),
                               "--runs", "3", "--samples", "3", "--output", str(self.root / output),
                               *map(str, arguments)], text=True, capture_output=True)

    def test_control_and_paired_run_preserve_every_sample(self):
        self.assertEqual(self.invoke("aa", "--control").returncode, 0)
        result = self.invoke("ab", "--candidate", self.binary, "--aa-control", self.root / "aa/summary.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(list((self.root / "ab").glob("*.stdout"))), 6)
        self.assertEqual(json.loads((self.root / "ab/summary.json").read_text())["status"], "PASS")

    def test_missing_control_cannot_certify_pass(self):
        self.assertEqual(self.invoke("ab", "--candidate", self.binary).returncode, 1)
        self.assertEqual(json.loads((self.root / "ab/summary.json").read_text())["status"], "UNQUALIFIED")

    def test_stale_control_cannot_certify_pass(self):
        self.assertEqual(self.invoke("aa", "--control").returncode, 0)
        self.binary.write_text(self.binary.read_text() + "# new binary revision\n")
        self.assertEqual(self.invoke("ab", "--candidate", self.binary, "--aa-control", self.root / "aa/summary.json").returncode, 1)
        self.assertFalse(json.loads((self.root / "ab/summary.json").read_text())["control_valid"])

    def test_existing_report_not_overwritten(self):
        self.assertEqual(self.invoke("aa", "--control").returncode, 0)
        saved = (self.root / "aa/summary.json").read_bytes()
        self.assertEqual(self.invoke("aa", "--control").returncode, 2)
        self.assertEqual((self.root / "aa/summary.json").read_bytes(), saved)

    def test_empty_filter_fails(self):
        self.assertEqual(self.invoke("aa", "--control", "--filter", "nonexistent").returncode, 2)

    def test_crash_preserves_stderr_and_no_pass_report(self):
        self.binary.write_text(f"#!{sys.executable}\nimport sys\n"
                               "if '--list' in sys.argv: print('u8/test/128x128/128x128')\n"
                               "else: print('intentional failure', file=sys.stderr); sys.exit(7)\n")
        self.assertEqual(self.invoke("aa", "--control").returncode, 2)
        self.assertIn("intentional failure", next((self.root / "aa").glob("*.stderr")).read_text())
        self.assertFalse((self.root / "aa/summary.json").exists())

    def test_timeout_does_not_produce_pass_report(self):
        self.binary.write_text(f"#!{sys.executable}\nimport sys, time\n"
                               "if '--list' in sys.argv: print('u8/test/128x128/128x128')\n"
                               "else: print('started', file=sys.stderr, flush=True); time.sleep(10)\n")
        self.assertEqual(self.invoke("aa", "--control", "--timeout", ".5").returncode, 2)
        self.assertIn("started", next((self.root / "aa").glob("*.stderr")).read_text())
        self.assertFalse((self.root / "aa/summary.json").exists())


if __name__ == "__main__":
    unittest.main()
