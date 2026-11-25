"""external preprocessing: skull strip, rigid registration to the baseline, n4, lesion-filled synthseg

needs on PATH: mri_synthstrip and mri_synthseg (freesurfer), flirt (fsl)
n4 needs the optional SimpleITK package
settings: those of the paper run, except the n4 mask: every brain voxel, not the sitk default 1..255
"""

import importlib.util
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import median_filter

from wmh_qc.containers import PreparedScan
from wmh_qc.loading import load_flair, load_labels, load_mask, on_grid, read_grid
from wmh_qc.tissue import WM_LABELS

SYNTHSTRIP = "mri_synthstrip"
FLIRT = "flirt"
SYNTHSEG = "mri_synthseg"
FLIRT_RIGID_MI = ("-dof", "6", "-cost", "mutualinfo", "-searchcost", "mutualinfo",
                  "-searchrx", "-10", "10", "-searchry", "-10", "10",
                  "-searchrz", "-10", "10", "-interp", "trilinear")
N4_ITERATIONS = (50, 50, 30, 20)
N4_CONVERGENCE = 0.001
FILL_SEED = 42
FILL_NOISE_FRACTION = 0.3         # fill noise sd as a fraction of the wm sd
FILL_MEDIAN_SIZE = 3

# prepared layout: out_dir/scan0 (baseline), scan1, ... each with these files
FLAIR_FILE = "flair_n4.nii.gz"
MASK_FILE = "mask.nii.gz"
LABELS_FILE = "synthseg.nii.gz"
UNFILLED_LABELS_FILE = "synthseg_unfilled.nii.gz"
# intermediates of `prepare`
BRAIN_FILE = "brain.nii.gz"
REGISTERED_FILE = "brain_registered.nii.gz"
MATRIX_FILE = "to_baseline.mat"
FILLED_FLAIR_FILE = "flair_filled.nii.gz"


def scan_dir(out_dir: Path, index: int) -> Path:
    return out_dir / f"scan{index}"


def check_tools() -> None:
    """every external tool and simpleitk present, so a run does not stop halfway"""
    missing = [tool for tool in (SYNTHSTRIP, FLIRT, SYNTHSEG) if shutil.which(tool) is None]
    if importlib.util.find_spec("SimpleITK") is None:
        missing.append('SimpleITK (pip install ".[prepare]")')
    if missing:
        raise FileNotFoundError(f"prepare needs: {', '.join(missing)}")


def _run(tool: str, args: list[str | Path]) -> None:
    executable = shutil.which(tool)
    if executable is None:
        raise FileNotFoundError(f"{tool} not found on PATH")
    try:
        subprocess.run([executable, *args], check=True)
    except subprocess.CalledProcessError as error:
        # the failed command line holds the input paths, clinics name folders after patients
        raise RuntimeError(f"{tool} failed with exit code {error.returncode}") from None


def skull_strip(image: Path, brain: Path) -> None:
    _run(SYNTHSTRIP, ["-i", image, "-o", brain])


def register_to_baseline(moving: Path, baseline: Path, out: Path, matrix: Path) -> None:
    _run(FLIRT, ["-in", moving, "-ref", baseline, "-out", out, "-omat", matrix, *FLIRT_RIGID_MI])


def transform_mask(mask: Path, baseline: Path, matrix: Path, out: Path) -> None:
    _run(FLIRT, ["-in", mask, "-ref", baseline, "-applyxfm", "-init", matrix,
                 "-interp", "nearestneighbour", "-out", out])


def n4_correct(image: Path, out: Path) -> None:
    """simpleitk n4 bias field correction over every brain voxel of a skull-stripped image"""
    import SimpleITK as sitk  # heavy optional dependency, only n4 needs it

    volume = sitk.ReadImage(str(image), sitk.sitkFloat32)
    mask = volume > 0
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(list(N4_ITERATIONS))
    corrector.SetConvergenceThreshold(N4_CONVERGENCE)
    sitk.WriteImage(corrector.Execute(volume, mask), str(out))


def synthseg(image: Path, labels: Path) -> None:
    """freesurfer synthseg. plain model, as the paper's lesion-filled segmentations"""
    _run(SYNTHSEG, ["--i", image, "--o", labels])


def synthseg_robust(image: Path, labels: Path) -> None:
    """freesurfer synthseg robust model, as the paper's unfilled segmentations"""
    _run(SYNTHSEG, ["--i", image, "--o", labels, "--robust"])


def fill_lesions(flair: np.ndarray, mask: np.ndarray, wm: np.ndarray) -> np.ndarray:
    """replace lesion voxels by noisy wm intensity so synthseg labels them as wm"""
    wm_values = flair[wm]
    wm_median = float(np.median(wm_values))
    wm_sd = float(np.std(wm_values))
    # legacy RandomState, not default_rng: same stream as the paper run's np.random.seed(42)
    noise = np.random.RandomState(FILL_SEED).normal(0, wm_sd * FILL_NOISE_FRACTION, int(mask.sum()))
    filled = flair.copy()
    filled[mask] = wm_median + noise.astype(np.float32)
    return np.where(mask, median_filter(filled, size=FILL_MEDIAN_SIZE), filled)


def segment_tissue(flair_path: Path, mask_path: Path, out_dir: Path) -> None:
    """filled and unfilled synthseg of a prepared flair. the unfilled one is the fill's wm reference"""
    grid = read_grid(flair_path)
    flair = load_flair(flair_path, grid)
    mask = load_mask(mask_path, grid)
    unfilled = out_dir / UNFILLED_LABELS_FILE
    synthseg_robust(flair_path, unfilled)
    wm = np.isin(load_labels(unfilled, grid), WM_LABELS) & ~mask & (flair > 0)
    image = nib.load(flair_path)
    filled = out_dir / FILLED_FLAIR_FILE
    nib.save(nib.Nifti1Image(fill_lesions(flair, mask, wm), image.affine, image.header), filled)
    synthseg(filled, out_dir / LABELS_FILE)


def prepare_series(raw_scans: list[Path], native_masks: list[Path], out_dir: Path) -> list[PreparedScan]:
    """baseline (first) scan defines the grid. later scans and their masks are moved onto it"""
    # a reused --out feeds stale scanN of an older, longer run into `run`. refuse, do not merge
    if scan_dir(out_dir, 0).exists():
        raise FileExistsError("the --out directory already holds prepared scans, use a new output directory")
    # refuse missing tools and bad inputs before minutes of external tools
    check_tools()
    for index, (raw, native_mask) in enumerate(zip(raw_scans, native_masks, strict=True)):
        grid = read_grid(raw)
        if not on_grid(nib.load(native_mask), grid):
            raise ValueError(f"scan {index}: {native_mask.name} does not have the shape and affine of {raw.name}")
        load_mask(native_mask, grid)
        load_flair(raw, grid)
    scan_dirs = [scan_dir(out_dir, index) for index in range(len(raw_scans))]
    for directory in scan_dirs:
        directory.mkdir(parents=True)
    baseline = scan_dirs[0] / BRAIN_FILE
    skull_strip(raw_scans[0], baseline)
    nib.save(nib.load(native_masks[0]), scan_dirs[0] / MASK_FILE)
    aligned = [baseline]
    for raw, native_mask, directory in zip(raw_scans[1:], native_masks[1:], scan_dirs[1:], strict=True):
        brain = directory / BRAIN_FILE
        registered = directory / REGISTERED_FILE
        matrix = directory / MATRIX_FILE
        skull_strip(raw, brain)
        register_to_baseline(brain, baseline, registered, matrix)
        transform_mask(native_mask, baseline, matrix, directory / MASK_FILE)
        aligned.append(registered)
    prepared: list[PreparedScan] = []
    for brain, directory in zip(aligned, scan_dirs, strict=True):
        n4_correct(brain, directory / FLAIR_FILE)
        segment_tissue(directory / FLAIR_FILE, directory / MASK_FILE, directory)
        prepared.append(PreparedScan(directory / FLAIR_FILE, directory / MASK_FILE,
                                     directory / LABELS_FILE, directory / UNFILLED_LABELS_FILE))
    return prepared


def read_prepared(out_dir: Path) -> list[PreparedScan]:
    """prepared scans scan0, scan1, ... in index order. the unfilled synthseg is optional"""
    prepared: list[PreparedScan] = []
    index = 0
    while scan_dir(out_dir, index).is_dir():
        directory = scan_dir(out_dir, index)
        required = [directory / name for name in (FLAIR_FILE, MASK_FILE, LABELS_FILE)]
        missing = [path for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{directory.name} incomplete, missing: {[path.name for path in missing]}")
        unfilled = directory / UNFILLED_LABELS_FILE
        prepared.append(PreparedScan(*required, unfilled if unfilled.exists() else None))
        index += 1
    if not prepared:
        raise FileNotFoundError(f"no {scan_dir(out_dir, 0).name} folder in the --prepared directory")
    return prepared
