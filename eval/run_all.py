"""Runs both eval scripts and merges their reports into eval/report.md.

The two underlying scripts remain independently runnable — this is a
convenience wrapper, not a replacement for them. Faithfulness costs real API
calls; retrieval quality is free and local.

Usage (from anywhere):
    python3 eval/run_all.py
"""
import os
import subprocess
import sys

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(_EVAL_DIR, "report.md")


def run() -> None:
    python = sys.executable
    subprocess.run([python, os.path.join(_EVAL_DIR, "retrieval_eval.py")], check=True)
    subprocess.run([python, os.path.join(_EVAL_DIR, "faithfulness_eval.py")], check=True)

    with open(os.path.join(_EVAL_DIR, "retrieval_report.md")) as f:
        retrieval_report = f.read()
    with open(os.path.join(_EVAL_DIR, "faithfulness_report.md")) as f:
        faithfulness_report = f.read()

    with open(REPORT_PATH, "w") as f:
        f.write(retrieval_report + "\n---\n\n" + faithfulness_report)

    print(f"\nMerged report written to {REPORT_PATH}")


if __name__ == "__main__":
    run()
