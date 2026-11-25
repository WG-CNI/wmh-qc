"""scan order check and pair bookkeeping of a series"""

import dataclasses
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from wmh_qc import series
from wmh_qc.containers import PreparedScan, ScanMeta
from wmh_qc.loading import Grid

UNREAD = PreparedScan(flair=Path("flair"), mask=Path("mask"), labels=Path("labels"), unfilled_labels=None)


class ImagesReached(Exception):
    """raised in place of image loading: the order check let the series through"""


def _refuse_images(path: Path) -> None:
    raise ImagesReached


@pytest.mark.parametrize("later_date", [datetime(2020, 1, 1), datetime(2021, 3, 4)])
def test_out_of_order_or_equal_dates_raise(same_meta: ScanMeta, later_date: datetime) -> None:
    """later scan dated before or on the earlier one raises before any image is read"""
    earlier = dataclasses.replace(same_meta, acquired=datetime(2021, 3, 4))
    later = dataclasses.replace(same_meta, acquired=later_date)
    with pytest.raises(ValueError, match="strictly increasing"):
        series.assess_series([UNREAD, UNREAD], [earlier, later])


def test_increasing_dates_pass(monkeypatch: pytest.MonkeyPatch, same_meta: ScanMeta) -> None:
    monkeypatch.setattr(series, "read_grid", _refuse_images)
    earlier = dataclasses.replace(same_meta, acquired=datetime(2020, 1, 1))
    later = dataclasses.replace(same_meta, acquired=datetime(2021, 3, 4))
    with pytest.raises(ImagesReached):
        series.assess_series([UNREAD, UNREAD], [earlier, later])


def test_unknown_date_warns_and_continues(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    same_meta: ScanMeta,
) -> None:
    """one unknown date: warning, no order check, images are read"""
    monkeypatch.setattr(series, "read_grid", _refuse_images)
    later = dataclasses.replace(same_meta, acquired=datetime(2000, 1, 1))
    with caplog.at_level(logging.WARNING), pytest.raises(ImagesReached):
        series.assess_series([UNREAD, UNREAD], [same_meta, later])
    assert "scan order not checked" in caplog.text


def test_three_scans_pair_previous_and_baseline(monkeypatch: pytest.MonkeyPatch, same_meta: ScanMeta) -> None:
    """three scans: 0->1, 1->2, then 0->2, each built from the right two scans"""
    files = [dataclasses.replace(UNREAD, flair=Path(f"scan{index}")) for index in range(3)]
    monkeypatch.setattr(series, "read_grid", lambda path: Grid((1, 1, 1), np.eye(4), (1.0, 1.0, 1.0)))
    monkeypatch.setattr(series, "load_scan", lambda scan_files, meta, grid: scan_files.flair.name)
    monkeypatch.setattr(series, "build_pair", lambda earlier, later, zooms_mm: (earlier, later))
    monkeypatch.setattr(series, "assess_pair", lambda pair: pair)
    pairs = series.assess_series(files, [same_meta] * 3)
    assert [(pair.earlier, pair.later) for pair in pairs] == [(0, 1), (1, 2), (0, 2)]
    assert [pair.result for pair in pairs] == [("scan0", "scan1"), ("scan1", "scan2"), ("scan0", "scan2")]
