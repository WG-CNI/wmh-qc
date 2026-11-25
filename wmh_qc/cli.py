"""command line: `wmh-qc prepare` for raw scans, `wmh-qc run` for prepared scans"""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from wmh_qc.metadata import read_dicom_meta, read_sidecar_meta
from wmh_qc.prepare import prepare_series, read_prepared
from wmh_qc.series import assess_series

LOGGER = logging.getLogger(__name__)


def _require(paths: list[Path]) -> None:
    """every input exists, checked before any work"""
    # library errors print the full path, and clinics name folders after patients
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"not found: {', '.join(missing)}")


def _prepare(args: argparse.Namespace) -> None:
    """raw scans and native masks -> prepared flair, mask and synthseg per scan"""
    raws = [Path(raw) for raw, _ in args.scan]
    masks = [Path(mask) for _, mask in args.scan]
    _require(raws + masks)
    for index, files in enumerate(prepare_series(raws, masks, args.out)):
        LOGGER.info("scan %d prepared: %s", index, files.flair.parent)


def _run(args: argparse.Namespace) -> None:
    """qc of all reported pairs, written as json"""
    # fail before minutes of computation, not when writing
    if not args.out.parent.is_dir():
        raise FileNotFoundError("the folder of --out does not exist")
    _require([args.prepared, *args.meta])
    metas = [read_dicom_meta(source) if source.is_dir() else read_sidecar_meta(source)
             for source in args.meta]
    records: list[dict[str, object]] = []
    for pair in assess_series(read_prepared(args.prepared), metas):
        kind = "consecutive" if pair.later == pair.earlier + 1 else "baseline"
        records.append({"earlier": pair.earlier, "later": pair.later, "pair": kind, **asdict(pair.result)})
    args.out.write_text(json.dumps(records, indent=1, default=float, allow_nan=False))
    for record in records:
        LOGGER.info("scan %d -> %d: dv %.2f ml, %s %s", record["earlier"], record["later"],
                    record["dv_ml"], record["verdict"], list(record["flags"]))
    LOGGER.info("done: %d pairs in %s. So long, and thanks for all the fish!", len(records), args.out.name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wmh-qc", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="skull strip, register to the first scan, n4, synthseg")
    prepare.add_argument("--scan", nargs=2, action="append", required=True, metavar=("RAW", "MASK"),
                         help="raw flair and its native wmh mask; repeat in acquisition order")
    prepare.add_argument("--out", type=Path, required=True, help="output directory")
    prepare.set_defaults(handler=_prepare)

    run = commands.add_parser("run", help="qc of prepared scans")
    run.add_argument("--prepared", type=Path, required=True,
                     help="directory written by `wmh-qc prepare` (scan0, scan1, ...)")
    run.add_argument("--meta", type=Path, action="append", required=True,
                     help="dicom directory or json sidecar of each scan, in acquisition order")
    run.add_argument("--out", type=Path, required=True, help="output json")
    run.set_defaults(handler=_run)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parser().parse_args()
    # refused input or a failed tool: one line, no traceback. anything else is a bug and keeps its traceback
    try:
        args.handler(args)
    except OSError as error:
        where = f": {Path(error.filename).name}" if error.filename else ""
        raise SystemExit(f"wmh-qc: error: {error.strerror or error}{where}") from None
    except (ValueError, RuntimeError) as error:
        raise SystemExit(f"wmh-qc: error: {error}") from None
