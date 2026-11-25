"""preprocessing: lesion fill, tool lookup, prepared layout"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from wmh_qc import prepare
from wmh_qc.prepare import (
    FLAIR_FILE,
    LABELS_FILE,
    MASK_FILE,
    UNFILLED_LABELS_FILE,
    fill_lesions,
    n4_correct,
    prepare_series,
    read_prepared,
    scan_dir,
    skull_strip,
)

TINY_SHAPE = (4, 4, 4)
RAW_WM = 300.0                    # raw scanner units


def test_fill_touches_only_lesions_and_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    flair = rng.normal(100.0, 5.0, (12, 12, 12)).astype(np.float32)
    mask = np.zeros(flair.shape, dtype=bool)
    mask[4:7, 4:7, 4:7] = True
    flair[mask] = 300.0
    wm = ~mask
    filled = fill_lesions(flair, mask, wm)
    np.testing.assert_array_equal(filled[~mask], flair[~mask])
    assert np.abs(filled[mask] - 100.0).max() < 10.0
    np.testing.assert_array_equal(filled, fill_lesions(flair, mask, wm))
    assert filled.dtype == np.float32


def test_missing_external_tool_fails_loudly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        skull_strip(tmp_path / "in.nii.gz", tmp_path / "out.nii.gz")


def test_missing_tools_are_listed_before_any_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="mri_synthstrip, flirt, mri_synthseg"):
        prepare.check_tools()


def test_failed_tool_error_holds_no_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """the failed command line names patient folders. only the tool and exit code are reported"""
    tool = tmp_path / "mri_synthstrip"
    tool.write_text("#!/bin/sh\nexit 3\n")
    tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(RuntimeError) as raised:
        skull_strip(tmp_path / "patient_name" / "in.nii.gz", tmp_path / "out.nii.gz")
    assert "exit code 3" in str(raised.value) and "patient_name" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_read_prepared_orders_numerically_and_unfilled_is_optional(tmp_path: Path) -> None:
    """scan2 comes before scan10. a missing unfilled synthseg reads as none"""
    for index in range(11):
        directory = scan_dir(tmp_path, index)
        directory.mkdir()
        for name in (FLAIR_FILE, MASK_FILE, LABELS_FILE):
            (directory / name).touch()
    (scan_dir(tmp_path, 0) / UNFILLED_LABELS_FILE).touch()
    prepared = read_prepared(tmp_path)
    assert [scan.flair.parent.name for scan in prepared][:3] == ["scan0", "scan1", "scan2"]
    assert prepared[10].flair.parent.name == "scan10"
    assert prepared[0].unfilled_labels is not None and prepared[1].unfilled_labels is None


def test_read_prepared_refuses_incomplete_scan(tmp_path: Path) -> None:
    directory = scan_dir(tmp_path, 0)
    directory.mkdir()
    (directory / FLAIR_FILE).touch()
    (directory / MASK_FILE).touch()
    with pytest.raises(FileNotFoundError):
        read_prepared(tmp_path)


def _fake_tools(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Path]]:
    """replace the external tools by file writers. returns the call log"""
    calls: list[tuple[str, Path]] = []

    def write(name: str, source: Path, target: Path) -> None:
        calls.append((name, source))
        target.touch()

    monkeypatch.setattr(prepare, "check_tools", lambda: None)
    monkeypatch.setattr(prepare, "skull_strip", lambda image, brain: write("strip", image, brain))
    monkeypatch.setattr(prepare, "register_to_baseline",
                        lambda moving, baseline, out, matrix: write("register", moving, out))
    monkeypatch.setattr(prepare, "transform_mask",
                        lambda mask, baseline, matrix, out: write("mask", mask, out))
    monkeypatch.setattr(prepare, "n4_correct", lambda image, out: write("n4", image, out))
    monkeypatch.setattr(prepare, "segment_tissue", lambda flair, mask, out_dir: [
        write("synthseg", flair, out_dir / name) for name in (LABELS_FILE, UNFILLED_LABELS_FILE)])
    return calls


def _inputs(tmp_path: Path, n_scans: int) -> tuple[list[Path], list[Path]]:
    raws = [tmp_path / f"raw{index}.nii.gz" for index in range(n_scans)]
    masks = [tmp_path / f"mask{index}.nii.gz" for index in range(n_scans)]
    for raw, mask in zip(raws, masks, strict=True):
        nib.save(nib.Nifti1Image(np.full(TINY_SHAPE, RAW_WM, dtype=np.float32), np.eye(4)), raw)
        nib.save(nib.Nifti1Image(np.zeros(TINY_SHAPE, dtype=np.uint8), np.eye(4)), mask)
    return raws, masks


def test_prepare_series_lays_out_every_scan_on_the_baseline(tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """three scans: later scans and only their masks are moved. run reads back the same layout"""
    calls = _fake_tools(monkeypatch)
    raws, masks = _inputs(tmp_path, 3)
    out = tmp_path / "prepared"
    prepared = prepare_series(raws, masks, out)
    assert prepared == read_prepared(out)
    assert [scan.flair.parent.name for scan in prepared] == ["scan0", "scan1", "scan2"]
    assert [source for name, source in calls if name == "strip"] == raws
    assert [source for name, source in calls if name == "mask"] == masks[1:]
    assert len([name for name, _ in calls if name == "register"]) == len(raws) - 1


def test_prepare_series_refuses_a_used_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """a second, shorter series into the same dir raises instead of leaving a stale scan behind"""
    _fake_tools(monkeypatch)
    out = tmp_path / "prepared"
    prepare_series(*_inputs(tmp_path, 3), out)
    with pytest.raises(FileExistsError):
        prepare_series(*_inputs(tmp_path, 2), out)
    assert len(read_prepared(out)) == 3


@pytest.mark.parametrize("change", ["shape", "affine"])
def test_prepare_refuses_a_mask_off_its_flair_grid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                   change: str) -> None:
    """a native mask on another grid than its raw flair raises with the scan index, before any tool runs"""
    calls = _fake_tools(monkeypatch)
    raws, masks = _inputs(tmp_path, 3)
    shapes = {"shape": (4, 4, 5), "affine": TINY_SHAPE}
    affines = {"shape": np.eye(4), "affine": np.diag([2.0, 2.0, 2.0, 1.0])}
    nib.save(nib.Nifti1Image(np.zeros(shapes[change], dtype=np.uint8), affines[change]), masks[1])
    with pytest.raises(ValueError, match="scan 1"):
        prepare_series(raws, masks, tmp_path / "prepared")
    assert calls == []


def test_prepare_refuses_a_probability_mask_before_any_tool(tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_tools(monkeypatch)
    raws, masks = _inputs(tmp_path, 2)
    nib.save(nib.Nifti1Image(np.linspace(0.0, 1.0, 64).reshape(TINY_SHAPE).astype(np.float32), np.eye(4)), masks[0])
    with pytest.raises(ValueError, match="binary"):
        prepare_series(raws, masks, tmp_path / "prepared")
    assert calls == []


def test_n4_corrects_every_brain_voxel_of_12_bit_data(tmp_path: Path) -> None:
    """raw 12-bit intensities, nothing in the old 1..255 mask: n4 still flattens a bias over the whole brain"""
    sitk = pytest.importorskip("SimpleITK")
    shape = (24, 24, 24)
    bias = np.linspace(0.8, 1.2, shape[0], dtype=np.float32)[:, None, None]
    brain = np.zeros(shape, dtype=np.float32)
    brain[2:-2, 2:-2, 2:-2] = 2000.0
    biased = brain * bias
    image = tmp_path / "brain.nii.gz"
    nib.save(nib.Nifti1Image(biased, np.eye(4)), image)
    out = tmp_path / "n4.nii.gz"
    n4_correct(image, out)
    corrected = sitk.GetArrayFromImage(sitk.ReadImage(str(out))).transpose()
    inside = brain > 0
    assert (corrected[~inside] == 0).all()
    assert corrected[inside].std() / corrected[inside].mean() < biased[inside].std() / biased[inside].mean()
