"""End-to-end coverage for `main.py --resume`: a real (non-mock) run against a
paid API can take many hours, so a crash partway through must not force
redoing already-completed (and already-paid-for) samples. These tests run the
actual CLI via subprocess with --mock so they stay fast and deterministic."""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_main(*args, cwd) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "main.py"), *args],
        cwd=cwd, capture_output=True, text=True, timeout=60,
    )


def _make_enriched_dataset(path: Path, n: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({
                "sample_id": f"S{i:04d}", "prompt": f"prompt-{i}", "category": "c",
                "category_macro": "cm", "target_claim": "claim", "value_axis": "v",
                "risk_type": "r", "expected_safe_behavior": "e",
            }, ensure_ascii=False) + "\n")


def _sample_ids(results_path: Path) -> list[str]:
    with open(results_path, encoding="utf-8") as f:
        return [json.loads(line)["sample_id"] for line in f if line.strip()]


def test_resume_skips_already_completed_samples(tmp_path):
    data_path = tmp_path / "data.jsonl"
    _make_enriched_dataset(data_path, n=5)
    out_path = tmp_path / "results.jsonl"
    bandit_path = tmp_path / "bandit.json"

    first = _run_main(
        "--mock", "--data", str(data_path), "--limit", "3",
        "--out", str(out_path), "--bandit-state", str(bandit_path),
        cwd=str(REPO_ROOT),
    )
    assert first.returncode == 0, first.stderr
    assert _sample_ids(out_path) == ["S0000", "S0001", "S0002"]

    second = _run_main(
        "--mock", "--data", str(data_path), "--limit", "5", "--resume",
        "--out", str(out_path), "--bandit-state", str(bandit_path),
        cwd=str(REPO_ROOT),
    )
    assert second.returncode == 0, second.stderr
    assert "3 sample(s) already" in second.stdout
    assert "2 remaining" in second.stdout
    assert _sample_ids(out_path) == ["S0000", "S0001", "S0002", "S0003", "S0004"]


def test_without_resume_overwrites_existing_output(tmp_path):
    data_path = tmp_path / "data.jsonl"
    _make_enriched_dataset(data_path, n=3)
    out_path = tmp_path / "results.jsonl"
    bandit_path = tmp_path / "bandit.json"

    _run_main(
        "--mock", "--data", str(data_path), "--limit", "3",
        "--out", str(out_path), "--bandit-state", str(bandit_path),
        cwd=str(REPO_ROOT),
    )
    assert _sample_ids(out_path) == ["S0000", "S0001", "S0002"]

    _run_main(
        "--mock", "--data", str(data_path), "--limit", "2",
        "--out", str(out_path), "--bandit-state", str(bandit_path),
        cwd=str(REPO_ROOT),
    )
    assert _sample_ids(out_path) == ["S0000", "S0001"]


def test_resume_with_no_existing_output_runs_everything(tmp_path):
    data_path = tmp_path / "data.jsonl"
    _make_enriched_dataset(data_path, n=2)
    out_path = tmp_path / "results.jsonl"
    bandit_path = tmp_path / "bandit.json"

    result = _run_main(
        "--mock", "--data", str(data_path), "--resume",
        "--out", str(out_path), "--bandit-state", str(bandit_path),
        cwd=str(REPO_ROOT),
    )

    assert result.returncode == 0, result.stderr
    assert _sample_ids(out_path) == ["S0000", "S0001"]
