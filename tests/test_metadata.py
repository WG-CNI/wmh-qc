"""metadata from dicom directories and json sidecars"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path

import pytest
from pydicom import dcmwrite
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import EnhancedMRImageStorage, ExplicitVRLittleEndian, MRImageStorage, generate_uid

from wmh_qc.containers import ScanMeta
from wmh_qc.metadata import read_dicom_meta, read_sidecar_meta

FLAIR = {"SeriesDescription": "t2_tirm_tra_dark-fluid", "EchoTime": 82.0, "RepetitionTime": 8500.0,
         "InversionTime": 2440.0}
RESCAN = {"SeriesDescription": "FLAIR rescan", "EchoTime": 105.0, "RepetitionTime": 8000.0, "InversionTime": 2370.7}
MPRAGE = {"SeriesDescription": "t1_mprage", "EchoTime": 3.0, "RepetitionTime": 2000.0, "InversionTime": 900.0}
CUSTOM_SIDECAR = {"te": 82, "tr": 8500, "ti": 2440, "field": 1.5, "vendor": "Siemens Healthineers",
                  "model": "MAGNETOM Altea", "slice_thickness": 3.0, "date": "2021-03-04T10:15:30"}
BIDS_SIDECAR = {"EchoTime": 0.082, "RepetitionTime": 8.5, "InversionTime": 2.44, "MagneticFieldStrength": 1.5,
                "Manufacturer": "Siemens Healthineers", "ManufacturersModelName": "MAGNETOM Altea",
                "SliceThickness": 3.0, "AcquisitionDateTime": "2021-03-04T10:15:30.000000",
                "SeriesDescription": "t2_tirm_tra_dark-fluid", "ConversionSoftware": "dcm2niix"}


def _write_dicom(path: Path, sop_class: str = MRImageStorage, **tags: object) -> None:
    """write a minimal mr dicom header: 1.5 t siemens plus the given tags"""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = sop_class
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    header = Dataset()
    header.file_meta = file_meta
    header.SOPClassUID = sop_class
    header.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    header.MagneticFieldStrength = 1.5
    header.Manufacturer = "SIEMENS "
    header.ManufacturerModelName = "SymphonyTim"
    header.SliceThickness = 3.0
    for keyword, value in tags.items():
        setattr(header, keyword, value)
    header.save_as(path, enforce_file_format=True)


def _write_series(directory: Path, files: int, **tags: object) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(files):
        _write_dicom(directory / f"img{index}.dcm", **tags)


def _write_json(path: Path, content: dict[str, object]) -> Path:
    path.write_text(json.dumps(content))
    return path


def test_majority_flair_series_wins(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """the flair series with most files wins and is logged, never the path. non-flair and non-dicom ignored"""
    session = tmp_path / "session"
    _write_series(session / "flair", 3, **FLAIR)
    _write_dicom(session / "flair" / "rescan.dcm", **RESCAN)
    _write_series(session / "t1", 5, **MPRAGE)
    (session / "notes.txt").write_text("not a dicom")
    with caplog.at_level(logging.WARNING):
        meta = read_dicom_meta(session)
    assert "2 flair series" in caplog.text and "t2_tirm_tra_dark-fluid" in caplog.text
    assert str(tmp_path) not in caplog.text
    assert (meta.te_ms, meta.tr_ms, meta.ti_ms) == (82.0, 8500.0, 2440.0)
    assert meta.field_t == 1.5 and meta.slice_thickness_mm == 3.0
    assert meta.vendor == "SIEMENS" and meta.model == "SymphonyTim"


def test_files_named_by_uid_or_ge_are_read(tmp_path: Path) -> None:
    """no suffix filter: uid-named and ge i123.MRDC.4 files are dicom too"""
    _write_dicom(tmp_path / "1.3.12.2.1107.5.2.30.25245.1", **FLAIR)
    _write_dicom(tmp_path / "i123.MRDC.4", **FLAIR)
    assert read_dicom_meta(tmp_path).te_ms == 82.0


def test_no_flair_gives_missing_metadata(tmp_path: Path) -> None:
    """no flair series gives nan, empty strings and no date"""
    _write_dicom(tmp_path / "a.dcm", **MPRAGE)
    meta = read_dicom_meta(tmp_path)
    assert math.isnan(meta.te_ms) and math.isnan(meta.field_t)
    assert meta.vendor == "" and meta.model == "" and meta.acquired is None


def test_enhanced_dicom_values_and_frame_weight(tmp_path: Path) -> None:
    """enhanced dicom: times from the functional groups, one file votes with all its frames"""
    shared = Dataset()
    shared.MRTimingAndRelatedParametersSequence = [Dataset()]
    shared.MRTimingAndRelatedParametersSequence[0].RepetitionTime = 5000.0
    shared.MRModifierSequence = [Dataset()]
    shared.MRModifierSequence[0].InversionTimes = [1800.0]
    shared.PixelMeasuresSequence = [Dataset()]
    shared.PixelMeasuresSequence[0].SliceThickness = 1.0
    frame = Dataset()
    frame.MREchoSequence = [Dataset()]
    frame.MREchoSequence[0].EffectiveEchoTime = 390.0
    _write_dicom(tmp_path / "enhanced", sop_class=EnhancedMRImageStorage, SeriesDescription="3D_FLAIR_SAG",
                 NumberOfFrames=4, SliceThickness=None, SharedFunctionalGroupsSequence=[shared],
                 PerFrameFunctionalGroupsSequence=[frame, frame, frame, frame])
    _write_series(tmp_path / "classic", 3, **FLAIR)
    meta = read_dicom_meta(tmp_path)
    assert (meta.te_ms, meta.tr_ms, meta.ti_ms, meta.slice_thickness_mm) == (390.0, 5000.0, 1800.0, 1.0)


def test_stir_and_t1_flair_rejected(tmp_path: Path) -> None:
    """tirm stir and t1 flair have short ti. the t2 flair wins despite fewer files"""
    _write_series(tmp_path / "stir", 5, SeriesDescription="tirm_cor", EchoTime=40.0, RepetitionTime=5000.0,
                  InversionTime=150.0)
    _write_series(tmp_path / "t1flair", 5, SeriesDescription="T1 FLAIR", EchoTime=9.0, RepetitionTime=2000.0,
                  InversionTime=860.0)
    _write_series(tmp_path / "t2flair", 2, **FLAIR)
    assert read_dicom_meta(tmp_path).ti_ms == 2440.0


@pytest.mark.parametrize("description", ["ax_dark_fluid", "DarkFluid", "T2 dark-fluid", "Sag FLAIR 3D"])
def test_description_punctuation_does_not_matter(tmp_path: Path, description: str) -> None:
    _write_dicom(tmp_path / "a.dcm", **(FLAIR | {"SeriesDescription": description}))
    assert read_dicom_meta(tmp_path).ti_ms == 2440.0


def test_derived_mpr_ignored(tmp_path: Path) -> None:
    """derived reformats never vote, even when they have more files"""
    _write_series(tmp_path / "mpr", 5, **(RESCAN | {"ImageType": ["DERIVED", "SECONDARY", "MPR"]}))
    _write_series(tmp_path / "flair", 2, **(FLAIR | {"ImageType": ["ORIGINAL", "PRIMARY", "M", "ND"]}))
    assert read_dicom_meta(tmp_path).te_ms == 82.0


def test_gauss_field_converted(tmp_path: Path) -> None:
    """a field above 100 is gauss, in dicom and in sidecars"""
    _write_dicom(tmp_path / "a.dcm", **(FLAIR | {"MagneticFieldStrength": 15000}))
    assert read_dicom_meta(tmp_path).field_t == 1.5
    sidecar = _write_json(tmp_path / "meta.json", CUSTOM_SIDECAR | {"field": 30000})
    assert read_sidecar_meta(sidecar).field_t == 3.0


def test_equal_multi_values_kept_different_dropped(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """repeated equal te is one te. different te values drop that file and the log names te"""
    _write_series(tmp_path / "equal", 1, **(FLAIR | {"EchoTime": [82.0, 82.0]}))
    _write_series(tmp_path / "different", 2, **(FLAIR | {"EchoTime": [82.0, 120.0]}))
    with caplog.at_level(logging.INFO):
        meta = read_dicom_meta(tmp_path)
    assert meta.te_ms == 82.0
    assert "(2 files): TE missing or multi-valued" in caplog.text


def test_implausible_or_missing_times_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """te above tr and a missing ti drop flair-looking series, the log names the field"""
    _write_dicom(tmp_path / "a.dcm", **(FLAIR | {"EchoTime": 9000.0}))
    no_ti = dict(FLAIR)
    del no_ti["InversionTime"]
    _write_dicom(tmp_path / "b.dcm", **no_ti)
    with caplog.at_level(logging.INFO):
        meta = read_dicom_meta(tmp_path)
    assert math.isnan(meta.te_ms)
    assert "TE implausible" in caplog.text and "TI missing" in caplog.text
    assert "no usable flair series" in caplog.text


def test_dicom_date_from_acquisition_else_study(tmp_path: Path) -> None:
    """combined acquisition datetime (zone dropped), else acquisition date and time, else study date"""
    _write_series(tmp_path / "acquisition", 1,
                  **(FLAIR | {"AcquisitionDate": "20210304", "AcquisitionTime": "101530.5", "StudyDate": "20200101"}))
    _write_series(tmp_path / "study", 1, **(FLAIR | {"StudyDate": "20200101"}))
    _write_series(tmp_path / "enhanced", 1, **(FLAIR | {"AcquisitionDateTime": "20210304101530.5+0100"}))
    assert read_dicom_meta(tmp_path / "enhanced").acquired == datetime(2021, 3, 4, 10, 15, 30, 500000)
    assert read_dicom_meta(tmp_path / "acquisition").acquired == datetime(2021, 3, 4, 10, 15, 30, 500000)
    assert read_dicom_meta(tmp_path / "study").acquired == datetime(2020, 1, 1)


def test_bids_sidecar_in_seconds_equals_custom_in_ms(tmp_path: Path) -> None:
    """dcm2niix keys in seconds give the same metadata as the custom keys in ms. extra bids keys ignored"""
    custom = read_sidecar_meta(_write_json(tmp_path / "custom.json", CUSTOM_SIDECAR))
    bids = read_sidecar_meta(_write_json(tmp_path / "bids.json", BIDS_SIDECAR))
    assert bids == custom
    assert custom == ScanMeta(te_ms=82.0, tr_ms=8500.0, ti_ms=2440.0, field_t=1.5, vendor="Siemens Healthineers",
                              model="MAGNETOM Altea", slice_thickness_mm=3.0,
                              acquired=datetime(2021, 3, 4, 10, 15, 30))


def test_sidecar_missing_fields_are_nan_or_empty(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "meta.json", {"te": 82, "tr": 8500, "ti": 2440, "vendor": "Siemens Healthineers",
                                                "model": None, "field": "nan"})
    meta = read_sidecar_meta(path)
    assert (meta.te_ms, meta.tr_ms, meta.ti_ms) == (82.0, 8500.0, 2440.0)
    assert meta.vendor == "Siemens Healthineers" and meta.model == ""
    assert math.isnan(meta.field_t) and math.isnan(meta.slice_thickness_mm) and meta.acquired is None


@pytest.mark.parametrize("content", [
    {},
    {"TE": 82, "TR": 8500},
    {"te": 82, "EchoTime": 0.082},
    {"te": 82, "tr": 8500, "comment": "typo or unit guess"},
    {"te": 9000, "tr": 8500},
    {"te": 82, "tr": 8500, "ti": 9000},
    {"te": 0, "tr": 8500},
    {"EchoTime": 0.082, "RepetitionTime": 8.5, "InversionTime": 0.0},
])
def test_bad_sidecar_refused(tmp_path: Path, content: dict[str, object]) -> None:
    """no known format, mixed or unknown keys, and implausible times raise"""
    with pytest.raises(ValueError, match="sidecar"):
        read_sidecar_meta(_write_json(tmp_path / "meta.json", content))


def test_sidecar_date_is_a_datetime(tmp_path: Path) -> None:
    """date only is midnight. a zoned date keeps its clock time, as dicom dates have no zone"""
    plain = _write_json(tmp_path / "plain.json", {"te": 82, "tr": 8500, "date": "2021-03-04"})
    zoned = _write_json(tmp_path / "zoned.json", {"te": 82, "tr": 8500, "date": "2021-03-04T10:15:30+01:00"})
    assert read_sidecar_meta(plain).acquired == datetime(2021, 3, 4)
    assert read_sidecar_meta(zoned).acquired == datetime(2021, 3, 4, 10, 15, 30)


def test_dicom_without_preamble_is_read(tmp_path: Path) -> None:
    """a raw dataset saved without preamble and file meta still counts"""
    header = Dataset()
    header.SOPClassUID = MRImageStorage
    header.MagneticFieldStrength = 1.5
    header.Manufacturer = "SIEMENS"
    for keyword, value in FLAIR.items():
        setattr(header, keyword, value)
    path = tmp_path / "IM0001"
    dcmwrite(path, header, implicit_vr=True, little_endian=True, enforce_file_format=False)
    assert path.read_bytes()[:2] == b"\x08\x00"
    assert read_dicom_meta(tmp_path).te_ms == 82.0


def test_non_dicom_file_starting_like_raw_dicom_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "notes").write_bytes(b"\x08\x00garbage that is not a dataset")
    assert math.isnan(read_dicom_meta(tmp_path).te_ms)


@pytest.mark.parametrize("content", [
    {"te": 0.082, "tr": 8.5, "ti": 2.44, "field": 1.5},
    {"EchoTime": 82, "RepetitionTime": 8500, "InversionTime": 2440},
])
def test_sidecar_times_in_the_wrong_unit_refused(tmp_path: Path, content: dict[str, object]) -> None:
    """custom times in s or bids times in ms are refused, not turned into a wrong mu"""
    with pytest.raises(ValueError, match="custom times are in ms, bids times in s"):
        read_sidecar_meta(_write_json(tmp_path / "meta.json", content))


def test_series_with_identical_timings_stay_apart(tmp_path: Path) -> None:
    """two series, same te/tr/ti: the larger one wins, its field strength is not mixed with the other"""
    _write_series(tmp_path / "a", 2, **FLAIR, SeriesInstanceUID=generate_uid(), MagneticFieldStrength=1.5)
    _write_series(tmp_path / "b", 3, **FLAIR, SeriesInstanceUID=generate_uid(), MagneticFieldStrength=3.0)
    assert read_dicom_meta(tmp_path).field_t == 3.0


def test_zero_field_strength_is_missing(tmp_path: Path) -> None:
    _write_dicom(tmp_path / "a.dcm", **FLAIR | {"MagneticFieldStrength": 0})
    assert math.isnan(read_dicom_meta(tmp_path).field_t)
