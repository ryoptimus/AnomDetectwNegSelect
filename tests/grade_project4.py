#!/usr/bin/env python3
"""Gradescope grader for CSE 568 Project 04.

The grader imports /autograder/submission/student_solution.py and tests:
  - process_data(file)
  - r_auc_tuple(file)

It writes /autograder/results/results.json using Gradescope's JSON format.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import signal
import sys
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

RESULTS_DIR = os.environ.get("GRADESCOPE_RESULTS_DIR", "/autograder/results")
SOURCE_DIR = os.environ.get("GRADESCOPE_SOURCE_DIR", "/autograder/source")
SUBMISSION_DIR = os.environ.get("GRADESCOPE_SUBMISSION_DIR", "/autograder/submission")

# Make local execution easy while preserving Gradescope behavior.
if not os.path.isdir(SOURCE_DIR):
    SOURCE_DIR = os.getcwd()
if not os.path.isdir(SUBMISSION_DIR):
    SUBMISSION_DIR = os.getcwd()

# Keep helper/data paths importable, but load the submitted file directly below.
# This prevents accidental fallback to a bundled source/student_solution.py.
for path in [os.path.join(SOURCE_DIR, "tests"), SOURCE_DIR, SUBMISSION_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)


def ensure_usercode_paths() -> None:
    """Create /usercode links when permissions allow, matching zyBooks paths."""
    try:
        os.makedirs("/usercode", exist_ok=True)
        os.makedirs("/usercode/negsel_src", exist_ok=True)
        links = {
            "/usercode/negsel.jar": os.path.join(SOURCE_DIR, "negsel.jar"),
            "/usercode/negsel_src/negsel.jar": os.path.join(SOURCE_DIR, "negsel.jar"),
            "/usercode/cardio_train.csv": os.path.join(SOURCE_DIR, "cardio_train.csv"),
            "/usercode/cardio_test.csv": os.path.join(SOURCE_DIR, "cardio_test.csv"),
            "/usercode/cardio.labels": os.path.join(SOURCE_DIR, "cardio.labels"),
        }
        for target, source in links.items():
            try:
                if os.path.lexists(target):
                    os.unlink(target)
                os.symlink(source, target)
            except Exception:
                pass
    except Exception:
        pass


ensure_usercode_paths()

# Run from the source directory so notebook-style relative paths work.
try:
    os.chdir(SOURCE_DIR)
except Exception:
    pass


def load_student_solution():
    """Load student_solution.py directly from the submission directory."""
    submission_file = os.path.join(SUBMISSION_DIR, "student_solution.py")
    if not os.path.isfile(submission_file):
        raise FileNotFoundError(
            "Expected a submitted file named student_solution.py in /autograder/submission"
        )
    spec = importlib.util.spec_from_file_location("student_solution", submission_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {submission_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["student_solution"] = module
    spec.loader.exec_module(module)
    return module

EXPECTED_TEST_ROWS = [
    "EDADDABADABFBEEAGFFCF",
    "EAACDADADACGAEHAGEFDF",
    "ECADDABBCACEBCFAGFFBJ",
    "EAAGFADAEACGAEGAGEECF",
    "ECADCABBCABEBCFAGFFBJ",
    "EDACDAABCACEBDEAGFFAJ",
    "EDACDAACCABFBEHAGFFBF",
    "EEACDAACCABHAGFAGGGAF",
    "EDACCABBCABGBEHAGFFBF",
    "EGADBAACCABGAFDAGGGAF",
]

EXPECTED_TRAIN_ROWS = [
    "EDAECEAACACIBJDBEDDAF",
    "EBAFCEAACACIBJCBEDDAF",
    "EBAFBEAADAEHAFGAEDDAJ",
    "EDAFAEAADADHAFFAEDDAJ",
    "ECBFBEACBACDDDCAEDDAJ",
    "EEEEAEACBABFBEBAEDEAJ",
    "DDICDECBCABGBGDAFCDEF",
    "DCHCCEEBCABGBGBAECDCF",
    "DDJDCEABCACIAHCAFCDDJ",
    "DBJDCEFBCABGAEDADCCCJ",
]

EXPECTED_AUC = {
    2: 0.7548143650242887,
    3: 0.7168459403192228,
    4: 0.2917071478140181,
    5: 0.0,
    6: 0.0,
    7: 0.0,
    8: 0.0,
    9: 0.0,
    10: 0.0,
}


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):  # type: ignore[no-untyped-def]
    raise TimeoutError("Timed out while running this check")


def run_with_timeout(func: Callable[[], None], seconds: int = 90) -> None:
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(seconds)
    try:
        func()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def numeric_value(value, *, label: str = "return value") -> float:
    """Accept DEAP-style (x,), one-item lists, numpy scalars, or plain floats."""
    if isinstance(value, (tuple, list)) and len(value) == 1:
        value = value[0]
    try:
        return float(value)
    except Exception as exc:
        raise AssertionError(f"{label} {value!r} is not numeric") from exc


def assert_auc_close(actual, expected: float, *, label: str) -> None:
    """Accept exact values, values rounded to 2 decimals, or tiny pandas-bin variations."""
    actual_float = numeric_value(actual, label=label)
    rounded_expected = round(expected, 2)

    if math.isclose(actual_float, expected, rel_tol=1e-5, abs_tol=1e-5):
        return
    if math.isclose(actual_float, rounded_expected, rel_tol=1e-9, abs_tol=0.0051):
        return
    # Different pandas versions can produce a few boundary-bin differences in
    # process_data(cardio_train.csv), so allow a small AUC tolerance.
    if math.isclose(actual_float, expected, rel_tol=0.0, abs_tol=0.01):
        return
    raise AssertionError(f"{label}: expected approximately {expected}, got {actual_float}")


def resolve_returned_path(path_value) -> str:
    if not isinstance(path_value, str):
        raise AssertionError(f"process_data should return a path string, got {path_value!r}")
    candidates = [path_value]
    if not os.path.isabs(path_value):
        candidates.extend([
            os.path.join(os.getcwd(), path_value),
            os.path.join(SOURCE_DIR, path_value),
            os.path.join("/usercode", path_value),
        ])
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise AssertionError(f"process_data returned {path_value!r}, but that file does not exist")


def read_processed_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def check_processed_file(path: str, expected_count: int, expected_rows: Sequence[str], label: str) -> None:
    lines = read_processed_lines(path)
    if len(lines) != expected_count:
        raise AssertionError(f"{label}: expected {expected_count} processed rows, got {len(lines)}")
    bad_rows = [i for i, line in enumerate(lines[: min(25, len(lines))]) if len(line) != 21]
    if bad_rows:
        first = bad_rows[0]
        raise AssertionError(f"{label}: row {first} should have length 21, got {len(lines[first])}")
    for i, line in enumerate(lines[: min(25, len(lines))]):
        invalid = sorted(set(line) - set("ABCDEFGHIJ"))
        if invalid:
            raise AssertionError(f"{label}: row {i} has invalid characters {invalid!r}")
    for index, expected in enumerate(expected_rows):
        if lines[index] != expected:
            raise AssertionError(
                f"{label}: row {index} expected {expected!r}, got {lines[index]!r}. "
                "Check that you excluded the sample and label columns and wrote no header/index."
            )


@dataclass
class GradedCheck:
    name: str
    max_score: float
    func: Callable[[object], None]


_r_auc_cache = None


def get_r_auc_tuples(student) -> List[Tuple[float, float]]:
    global _r_auc_cache
    if _r_auc_cache is None:
        if not hasattr(student, "r_auc_tuple"):
            raise AssertionError("student_solution.py must define r_auc_tuple(file)")
        raw = student.r_auc_tuple("cardio_test.csv")
        if not isinstance(raw, (list, tuple)):
            raise AssertionError(f"r_auc_tuple should return a list of tuples, got {type(raw).__name__}")
        parsed = []
        for item in raw:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise AssertionError(f"Each r_auc_tuple item should be a pair (r, auc), got {item!r}")
            parsed.append((numeric_value(item[0], label="r value"), numeric_value(item[1], label="AUC value")))
        _r_auc_cache = parsed
    return _r_auc_cache


def tuple_dict(student) -> Dict[int, float]:
    tuples = get_r_auc_tuples(student)
    result: Dict[int, float] = {}
    for r_value, auc_value in tuples:
        r_int = int(r_value)
        if not math.isclose(r_value, r_int, rel_tol=0.0, abs_tol=1e-9):
            raise AssertionError(f"r values should be integers, got {r_value!r}")
        result[r_int] = auc_value
    return result


def check_process_test(student) -> None:
    if not hasattr(student, "process_data"):
        raise AssertionError("student_solution.py must define process_data(file)")
    output_path = resolve_returned_path(student.process_data("cardio_test.csv"))
    check_processed_file(output_path, 831, EXPECTED_TEST_ROWS, "process_data(cardio_test.csv)")


def check_process_train(student) -> None:
    if not hasattr(student, "process_data"):
        raise AssertionError("student_solution.py must define process_data(file)")
    output_path = resolve_returned_path(student.process_data("cardio_train.csv"))
    check_processed_file(output_path, 1000, EXPECTED_TRAIN_ROWS, "process_data(cardio_train.csv)")


def check_r_auc_structure(student) -> None:
    tuples = get_r_auc_tuples(student)
    if len(tuples) != 9:
        raise AssertionError(f"r_auc_tuple should return 9 tuples for r=2..10, got {len(tuples)}")
    r_values = [int(r) for r, _ in tuples]
    if r_values != list(range(2, 11)):
        raise AssertionError(f"Expected r values [2, 3, ..., 10] in order, got {r_values}")


def check_r2_auc(student) -> None:
    values = tuple_dict(student)
    assert_auc_close(values.get(2), EXPECTED_AUC[2], label="AUC for r=2")


def check_r3_auc(student) -> None:
    values = tuple_dict(student)
    assert_auc_close(values.get(3), EXPECTED_AUC[3], label="AUC for r=3")


def check_r4_auc(student) -> None:
    values = tuple_dict(student)
    assert_auc_close(values.get(4), EXPECTED_AUC[4], label="AUC for r=4")


def check_high_r_zero_auc(student) -> None:
    values = tuple_dict(student)
    for r in range(5, 11):
        assert_auc_close(values.get(r), EXPECTED_AUC[r], label=f"AUC for r={r}")


def build_checks() -> List[GradedCheck]:
    return [
        GradedCheck("Q1 process_data cardio_test.csv", 12, check_process_test),
        GradedCheck("Q1 process_data cardio_train.csv", 13, check_process_train),
        GradedCheck("Q3 r_auc_tuple structure", 10, check_r_auc_structure),
        GradedCheck("Q3 AUC for r=2 sample", 15, check_r2_auc),
        GradedCheck("Q3 AUC for r=3", 15, check_r3_auc),
        GradedCheck("Q3 AUC for r=4", 15, check_r4_auc),
        GradedCheck("Q3 AUC for r=5 through r=10", 20, check_high_r_zero_auc),
    ]


def result_entry(name: str, score: float, max_score: float, output: str) -> dict:
    return {
        "name": name,
        "score": score,
        "max_score": max_score,
        "output": output,
        "visibility": "visible",
    }


def write_results(payload: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> int:
    tests = []
    total_score = 0.0
    max_score = 100.0

    try:
        student = load_student_solution()
    except Exception:
        output = (
            "Could not import student_solution.py. Remove unguarded notebook/demo code that fails "
            "during import and make sure the file is named student_solution.py.\n\n"
        )
        output += traceback.format_exc(limit=20)
        payload = {
            "score": 0,
            "visibility": "visible",
            "stdout_visibility": "visible",
            "output": output,
            "tests": [result_entry("Import student_solution.py", 0, max_score, output)],
        }
        write_results(payload)
        print(output)
        return 0

    for check in build_checks():
        try:
            run_with_timeout(lambda check=check: check.func(student), seconds=120)
        except Exception:
            output = traceback.format_exc(limit=8)
            tests.append(result_entry(check.name, 0, check.max_score, output))
            print(f"FAIL: {check.name}\n{output}")
        else:
            tests.append(result_entry(check.name, check.max_score, check.max_score, "Passed"))
            total_score += check.max_score
            print(f"PASS: {check.name}")

    payload = {
        "score": total_score,
        "visibility": "visible",
        "stdout_visibility": "visible",
        "output": f"Project 04 autograder score: {total_score:.1f} / {max_score:.1f}",
        "tests": tests,
    }
    write_results(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
