"""cli output: qc.json is strict json"""

import dataclasses
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from wmh_qc import cli
from wmh_qc.assess import assess_pair
from wmh_qc.containers import PairData, PreparedScan, ScanMeta
from wmh_qc.series import SeriesPair


def _refuse_constant(name: str) -> None:
    """json parse hook: nan and infinity are not json"""
    raise ValueError(f"non-json constant {name}")


def test_no_call_severity_is_written_as_null(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    same_meta: ScanMeta,
    make_pair: Callable[...,
    PairData],
) -> None:
    no_call = assess_pair(make_pair(meta2=dataclasses.replace(same_meta, te_ms=math.nan)))

    def fake_series(prepared: list[PreparedScan], metas: list[ScanMeta]) -> list[SeriesPair]:
        return [SeriesPair(0, 1, no_call)]

    monkeypatch.setattr(cli, "read_prepared", lambda out_dir: [])
    monkeypatch.setattr(cli, "assess_series", fake_series)
    sidecar = tmp_path / "meta.json"
    sidecar.write_text(json.dumps({"te": None}))
    out = tmp_path / "qc.json"
    argv = ["wmh-qc", "run", "--prepared", str(tmp_path), "--meta", str(sidecar), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    records = json.loads(out.read_text(), parse_constant=_refuse_constant)
    assert records[0]["severity"] is None
    assert records[0]["verdict"] == no_call.verdict


def test_missing_output_folder_fails_before_any_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def never(*args: object) -> None:
        raise AssertionError("qc started although --out cannot be written")

    monkeypatch.setattr(cli, "read_prepared", never)
    argv = ["wmh-qc", "run", "--prepared", str(tmp_path), "--meta", str(tmp_path),
            "--out", str(tmp_path / "missing" / "qc.json")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="wmh-qc: error: the folder of --out does not exist"):
        cli.main()


def test_missing_input_names_the_file_never_its_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """one line without traceback, and no patient folder in it"""
    sidecar = tmp_path / "Doe_Jane" / "meta.json"
    argv = ["wmh-qc", "run", "--prepared", str(tmp_path), "--meta", str(sidecar), "--out", str(tmp_path / "qc.json")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as stop:
        cli.main()
    assert str(stop.value) == "wmh-qc: error: not found: meta.json"
