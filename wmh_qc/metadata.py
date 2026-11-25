"""flair acquisition metadata from a dicom directory or a json sidecar"""

import json
import logging
import math
import struct
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, time
from pathlib import Path

import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.multival import MultiValue
from pydicom.valuerep import DA, DT, TM

from wmh_qc.containers import ScanMeta
from wmh_qc.transition import name_key

LOGGER = logging.getLogger(__name__)

# matched against series text reduced to lowercase letters and digits
FLAIR_KEYWORDS = ("flair", "darkfluid", "tirm")
DESCRIPTION_TAGS = ("SeriesDescription", "ProtocolName", "SequenceName")
# numeric dicom keyword per raw value key
NUMERIC_TAGS = {"te": "EchoTime", "tr": "RepetitionTime", "ti": "InversionTime",
                "slice_thickness": "SliceThickness"}
NON_ORIGINAL_IMAGE_TYPES = {"DERIVED", "SECONDARY"}
# acquisition, then series, then study date with its time
DATE_TIME_TAGS = (("AcquisitionDate", "AcquisitionTime"), ("SeriesDate", "SeriesTime"), ("StudyDate", "StudyTime"))
# enhanced dicom keeps these in functional groups: top-level keyword -> (group macro, keyword inside)
FUNCTIONAL_GROUP_TAGS = {
    "EchoTime": ("MREchoSequence", "EffectiveEchoTime"),
    "RepetitionTime": ("MRTimingAndRelatedParametersSequence", "RepetitionTime"),
    "InversionTime": ("MRModifierSequence", "InversionTimes"),
    "SliceThickness": ("PixelMeasuresSequence", "SliceThickness"),
}
# shared groups first, then the first frame of the per-frame groups
FUNCTIONAL_GROUPS = ("SharedFunctionalGroupsSequence", "PerFrameFunctionalGroupsSequence")
READ_TAGS = (*FUNCTIONAL_GROUP_TAGS, *FUNCTIONAL_GROUPS, *DESCRIPTION_TAGS,
             *(tag for pair in DATE_TIME_TAGS for tag in pair),
             "MagneticFieldStrength", "Manufacturer", "ManufacturerModelName", "ImageType", "NumberOfFrames",
             "SeriesInstanceUID", "AcquisitionDateTime")
# csf null: ti = t1_csf ln(2 / (1 + exp(-tr / t1_csf))), 1500 ms or more for every t2-flair.
# t1-flair (~860 ms) and stir (~150 ms) sit far below
MIN_FLAIR_TI_MS = 1200.0
# the longest clinical flair tr is about 11 s. a bids sidecar written in ms and read as s lands far above
MAX_TR_MS = 30000.0
# dicom saved without the 128-byte preamble starts with a group 0002 or 0008 tag, little endian
RAW_DICOM_STARTS = (b"\x02\x00", b"\x08\x00")
# no clinical magnet comes near 100 t, a larger header value is gauss
MAX_FIELD_T = 100.0
GAUSS_PER_TESLA = 10000.0
MS_PER_S = 1000.0
SERIES_KEY_DECIMALS = 1           # te/tr/ti rounding when a file has no series uid
SERIES_KEY_DESCRIPTION_CHARS = 64
MISSING_TEXT = ("none", "nan")

# custom sidecar: times in ms, field in t, optional iso "date"
CUSTOM_KEYS = {"te", "tr", "ti", "field", "vendor", "model", "slice_thickness", "date"}
CUSTOM_TIMING_KEYS = ("te", "tr", "ti")
# bids / dcm2niix sidecar key -> custom key. times in s there, other keys of that open format ignored
BIDS_KEYS = {
    "EchoTime": "te", "RepetitionTime": "tr", "InversionTime": "ti", "MagneticFieldStrength": "field",
    "Manufacturer": "vendor", "ManufacturersModelName": "model", "SliceThickness": "slice_thickness",
    "AcquisitionDateTime": "date",
}

# rounded te, tr, ti and the description prefix of one series
# series instance uid, else rounded te, tr, ti and the description prefix
SeriesKey = str | tuple[float, float, float, str]


def _values(value: object) -> list[object]:
    return list(value) if isinstance(value, MultiValue) else [value]


def _number(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _scalar(value: object) -> float:
    """single number of a header value. nan when absent or when its values differ"""
    numbers = {_number(item) for item in _values(value)}
    return numbers.pop() if len(numbers) == 1 else math.nan


def _text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in MISSING_TEXT else text


def _meta(values: Mapping[str, object], acquired: datetime | None) -> ScanMeta:
    """scan metadata from raw values keyed te, tr, ti, field, vendor, model, slice_thickness. gauss field to t"""
    field = _number(values.get("field"))
    # anonymisers write 0 for an unknown field
    if field <= 0:
        field = math.nan
    return ScanMeta(te_ms=_number(values.get("te")), tr_ms=_number(values.get("tr")),
                    ti_ms=_number(values.get("ti")),
                    field_t=field / GAUSS_PER_TESLA if field > MAX_FIELD_T else field,
                    vendor=_text(values.get("vendor")), model=_text(values.get("model")),
                    slice_thickness_mm=_number(values.get("slice_thickness")), acquired=acquired)


def _time_problem(meta: ScanMeta) -> str:
    """why the acquisition times rule out a t2-flair. "" when they do not"""
    # nan compares false, unknown times never fail here
    checks = (("TR", meta.tr_ms <= 0 or meta.tr_ms > MAX_TR_MS),
              ("TE", meta.te_ms <= 0 or meta.te_ms >= meta.tr_ms),
              ("TI", meta.ti_ms <= 0 or meta.ti_ms >= meta.tr_ms))
    implausible = next((name for name, bad in checks if bad), "")
    if implausible:
        return f"{implausible} implausible, need 0 < TE < TR, 0 < TI < TR and TR up to {MAX_TR_MS:.0f} ms"
    if meta.ti_ms < MIN_FLAIR_TI_MS:
        return f"TI below {MIN_FLAIR_TI_MS:.0f} ms does not null csf (stir or t1-flair)"
    return ""


def read_sidecar_meta(path: Path) -> ScanMeta:
    """scan metadata from a custom json sidecar (ms) or a bids / dcm2niix sidecar (s)"""
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("sidecar must be a json object")
    custom = raw.keys() & CUSTOM_KEYS
    bids = raw.keys() & BIDS_KEYS.keys()
    if custom and bids:
        raise ValueError(f"sidecar mixes custom keys {sorted(custom)} with bids keys {sorted(bids)}")
    if not (custom or bids):
        raise ValueError(f"sidecar has neither custom keys {sorted(CUSTOM_KEYS)} nor bids keys {sorted(BIDS_KEYS)}")
    unknown = raw.keys() - CUSTOM_KEYS
    if custom and unknown:
        raise ValueError(f"unknown custom sidecar keys {sorted(unknown)}")
    values = dict(raw) if custom else {BIDS_KEYS[key]: value for key, value in raw.items() if key in BIDS_KEYS}
    if bids:
        values.update({key: _number(values.get(key)) * MS_PER_S for key in CUSTOM_TIMING_KEYS})
    date = values.get("date")
    acquired = None if date is None else datetime.fromisoformat(date)
    # dicom dates carry no zone. a zoned sidecar date keeps its wall-clock time so both compare
    if acquired is not None and acquired.tzinfo is not None:
        acquired = acquired.replace(tzinfo=None)
    meta = _meta(values, acquired)
    problem = _time_problem(meta)
    if problem:
        raise ValueError(f"sidecar {problem}. custom times are in ms, bids times in s")
    return meta


def _read_header(path: Path) -> pydicom.Dataset | None:
    """header of a dicom file, with or without the preamble. none for anything else"""
    try:
        with path.open("rb") as file:
            raw = file.read(len(RAW_DICOM_STARTS[0])) in RAW_DICOM_STARTS
        return pydicom.dcmread(path, stop_before_pixels=True, specific_tags=list(READ_TAGS), force=raw)
    except (InvalidDicomError, OSError, EOFError, ValueError, struct.error):
        return None


def _header_value(header: pydicom.Dataset, keyword: str) -> object:
    """tag value at top level, else from the enhanced functional groups. none when absent"""
    top = header.get(keyword)
    if top is not None:
        return top
    macro, inner = FUNCTIONAL_GROUP_TAGS[keyword]
    for groups_keyword in FUNCTIONAL_GROUPS:
        groups = header.get(groups_keyword)
        items = groups[0].get(macro) if groups else None
        value = items[0].get(inner) if items else None
        if value is not None:
            return value
    return None


def _acquired(header: pydicom.Dataset) -> datetime | None:
    """acquisition, else series, else study date and time. none when no date is recorded"""
    # enhanced dicom keeps only the combined tag. the zone is dropped like for sidecar dates
    combined = header.get("AcquisitionDateTime")
    if combined:
        return DT(combined).replace(tzinfo=None)
    for date_tag, time_tag in DATE_TIME_TAGS:
        date = header.get(date_tag)
        if date:
            clock = header.get(time_tag)
            return datetime.combine(DA(date), TM(clock) if clock else time())
    return None


def _flair_like(header: pydicom.Dataset) -> bool:
    """original image whose series text names a flair"""
    image_type = {str(value).upper() for value in _values(header.get("ImageType"))}
    if image_type & NON_ORIGINAL_IMAGE_TYPES:
        return False
    # per tag, so no keyword is glued together across two tags
    text = " ".join(name_key(str(header.get(tag, ""))) for tag in DESCRIPTION_TAGS)
    return any(keyword in text for keyword in FLAIR_KEYWORDS)


def _header_meta(header: pydicom.Dataset) -> ScanMeta:
    values = {key: _scalar(_header_value(header, tag)) for key, tag in NUMERIC_TAGS.items()}
    values["field"] = header.get("MagneticFieldStrength")
    values["vendor"] = header.get("Manufacturer")
    values["model"] = header.get("ManufacturerModelName")
    return _meta(values, _acquired(header))


def _drop_reason(meta: ScanMeta) -> str:
    """why a flair-looking image cannot vote. "" when it can"""
    for name, value in (("TE", meta.te_ms), ("TR", meta.tr_ms), ("TI", meta.ti_ms)):
        if math.isnan(value):
            return f"{name} missing or multi-valued with different values"
    return _time_problem(meta)


def read_dicom_meta(directory: Path) -> ScanMeta:
    """flair metadata of one session: the flair series with the most frames wins"""
    counts: Counter[SeriesKey] = Counter()
    # first file of each series: its metadata and its description for the log
    firsts: dict[SeriesKey, tuple[ScanMeta, str]] = {}
    dropped: Counter[str] = Counter()
    for path in sorted(directory.rglob("*")):
        header = _read_header(path) if path.is_file() else None
        if header is None or not _flair_like(header):
            continue
        meta = _header_meta(header)
        reason = _drop_reason(meta)
        if reason:
            dropped[reason] += 1
            continue
        description = " ".join(str(header.get(tag, "")) for tag in DESCRIPTION_TAGS).lower()
        # two series with the same timings are still two series, e.g. a rescan or another field strength
        key = _text(header.get("SeriesInstanceUID")) or (
            round(meta.te_ms, SERIES_KEY_DECIMALS), round(meta.tr_ms, SERIES_KEY_DECIMALS),
            round(meta.ti_ms, SERIES_KEY_DECIMALS), description[:SERIES_KEY_DESCRIPTION_CHARS])
        counts[key] += int(header.get("NumberOfFrames") or 1)
        firsts.setdefault(key, (meta, description))
    # never log the path, clinics name folders after patients
    for reason, files in dropped.items():
        LOGGER.info("flair-looking series dropped (%d files): %s", files, reason)
    if not counts:
        LOGGER.warning("no usable flair series found. metadata missing")
        return _meta({}, None)
    # tie: first series in sorted path order wins, max guarantees it
    winner = max(counts, key=counts.get)
    chosen, description = firsts[winner]
    if len(counts) > 1:
        LOGGER.warning("%d flair series found. majority vote by frame count picks %r, TE %.1f TR %.1f TI %.1f ms",
                       len(counts), description.strip(), chosen.te_ms, chosen.tr_ms, chosen.ti_ms)
    return chosen
