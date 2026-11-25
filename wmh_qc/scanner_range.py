"""range of the supplied-mask volume change that a pure scanner change could produce (eq. 7)

each mask is re-thresholded after the physics mu contrast shift and after the sharpness match
the voxel-count change is the scanner effect
both scans estimate the same effect, so it is averaged
threshold noise: +-1.96 sd, spatially correlated voxels
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, median_filter
from scipy.special import ndtr

from wmh_qc.constants import DEFAULT_BLUR_RATIO
from wmh_qc.containers import PairData, Zooms
from wmh_qc.spacing import axis_steps, voxel_ml

CROP_PAD_MM = 8.0              # 16 voxels at 0.5 mm iso
REGION_MM = 2.0                # threshold re-estimation region around the mask
GAP_MM = 2.0                   # normal-appearing wm starts this far from the mask
PSF_NATIVE_MM = 0.4            # same constant as the cache builder's psf match
SHARPER_LATER_RATIO = 1.10     # blur_ratio above: scan 2 sharper, blur it
SHARPER_EARLIER_RATIO = 0.90   # blur_ratio below: scan 1 sharper, blur it
EMPTY_MASK_THRESHOLD_EPS_LOG = 1e-6   # no mask voxel in the region: threshold above everything
RESIDUAL_MEDIAN_RADIUS_MM = 1.0   # high-pass median window 5 voxels at 0.5 mm iso
FWHM_PER_SIGMA = 2.355
RHO_MIN, RHO_MAX = 1e-3, 0.999
Z_95 = 1.96


@dataclass(frozen=True)
class ScannerRange:
    """supplied-mask change and the scanner-only range around it, in ml"""

    dv_ml: float
    lo_ml: float
    hi_ml: float
    scanner_ml: tuple[float, float]    # contrast + blur effect, lowest and highest over the mu band
    contrast_ml: tuple[float, float]
    blur_ml: float
    noise_sd_ml: float


# no lesion at either scan: nothing to re-threshold
NO_LESION_RANGE = ScannerRange(dv_ml=0.0, lo_ml=0.0, hi_ml=0.0, scanner_ml=(0.0, 0.0), contrast_ml=(0.0, 0.0),
                               blur_ml=0.0, noise_sd_ml=0.0)


@dataclass(frozen=True)
class CroppedScan:
    """one scan's log image and mask on the crop, with the gaussian sigma that matches its sharpness"""

    image: np.ndarray
    mask: np.ndarray
    blur_sigma_vox: tuple[float, ...] | None


@dataclass(frozen=True)
class ScanEffect:
    """scanner effect seen from one scan"""

    contrast_ml: tuple[float, ...]     # aligned with the shifts passed in
    blur_ml: float
    noise_var_vox: float


def mask_change_ml(pair: PairData) -> float:
    return float(pair.m2.sum() - pair.m1.sum()) * voxel_ml(pair.zooms_mm)


def blur_sigmas_vox(blur_ratio: float, zooms_mm: Zooms
                    ) -> tuple[tuple[float, ...] | None, tuple[float, ...] | None]:
    """gaussian sigma per axis that blurs the sharper scan to the other. none for the blurrier one"""
    if blur_ratio < SHARPER_EARLIER_RATIO:
        sigma_mm = PSF_NATIVE_MM * np.sqrt(max(1.0 / blur_ratio ** 2 - 1, 0))
        return tuple(sigma_mm / float(zoom) for zoom in zooms_mm), None
    if blur_ratio > SHARPER_LATER_RATIO:
        sigma_mm = PSF_NATIVE_MM * np.sqrt(max(blur_ratio ** 2 - 1, 0))
        return None, tuple(sigma_mm / float(zoom) for zoom in zooms_mm)
    return None, None


def mask_crop(mask: np.ndarray, zooms_mm: Zooms) -> tuple[slice, ...]:
    """bounding box of a mask padded by CROP_PAD_MM per axis"""
    coords = np.nonzero(mask)
    pads = axis_steps(CROP_PAD_MM, zooms_mm)
    return tuple(slice(max(0, c.min() - pad), c.max() + 1 + pad) for c, pad in zip(coords, pads, strict=True))


def l1_ball(radius_vox: tuple[int, ...]) -> np.ndarray:
    """diamond structuring element with per-axis radii (iterated 6-neighbour dilation when isotropic)"""
    # voxel offset d is inside when sum |d_i| / r_i <= 1. both sides times prod(r): exact integers,
    # no float rounding decides a boundary voxel
    radius_product = int(np.prod(radius_vox))
    offsets = np.ogrid[tuple(slice(-radius, radius + 1) for radius in radius_vox)]
    scaled_sum = sum(np.abs(offset) * (radius_product // radius)
                     for offset, radius in zip(offsets, radius_vox, strict=True))
    return scaled_sum <= radius_product


def dilate_mm(mask: np.ndarray, radius_mm: float, zooms_mm: Zooms) -> np.ndarray:
    return binary_dilation(mask, structure=l1_ball(axis_steps(radius_mm, zooms_mm)))


def smoothness(image: np.ndarray, region: np.ndarray, zooms_mm: Zooms) -> float:
    """voxels per independent noise sample, from lag-1 autocorrelation of the high-pass residual"""
    # odd window per axis. a thick-slice axis may get size 1, no filtering along it
    window = tuple(2 * round(RESIDUAL_MEDIAN_RADIUS_MM / float(zoom)) + 1 for zoom in zooms_mm)
    residual = image - median_filter(image, size=window)
    voxels_per_sample = 1.0
    for axis in range(residual.ndim):
        last = residual.shape[axis] - 1
        lead = np.take(residual, range(last), axis=axis)
        lag = np.take(residual, range(1, last + 1), axis=axis)
        pairs = np.take(region, range(last), axis=axis) & np.take(region, range(1, last + 1), axis=axis)
        rho = float(np.clip(np.corrcoef(lead[pairs], lag[pairs])[0, 1], RHO_MIN, RHO_MAX))
        sigma_vox = np.sqrt(-1.0 / (4.0 * np.log(rho)))
        voxels_per_sample *= max(1.0, FWHM_PER_SIGMA * sigma_vox)
    return voxels_per_sample


def scan_effect(scan: CroppedScan, domain: np.ndarray, zooms_mm: Zooms,
                shifts: list[float], noise_sd_log: np.float64) -> ScanEffect:
    """mask-volume change of one scan under each contrast shift and the sharpness match"""
    # no lesion voxel in the domain (empty mask, or lesions only in excluded tissue): nothing to re-threshold.
    # this scan adds 0, so its half of the contrast average is 0 and it gives no blur term
    if not (scan.mask & domain).any():
        return ScanEffect(contrast_ml=(0.0,) * len(shifts), blur_ml=0.0, noise_var_vox=0.0)
    voxel_volume_ml = voxel_ml(zooms_mm)
    region = dilate_mm(scan.mask, REGION_MM, zooms_mm) & domain
    nawm = domain & ~dilate_mm(scan.mask, GAP_MM, zooms_mm)
    lesion_log = float(np.median(scan.image[scan.mask & domain]))
    wm_log = float(np.median(scan.image[nawm]))
    if lesion_log <= wm_log:
        raise ValueError("supplied lesions are not brighter than the white matter around them; "
                         "check that each mask belongs to its flair")
    values = scan.image[region]
    keep = int((scan.mask & region).sum())
    if keep > 0:
        threshold = float(np.sort(values)[values.size - keep])
    else:
        threshold = float(values.max()) + EMPTY_MASK_THRESHOLD_EPS_LOG
    base = int((values >= threshold).sum())
    lesion_weight = np.clip((values - wm_log) / (lesion_log - wm_log), 0, 1)
    contrast_ml = tuple((int((values + shift * lesion_weight >= threshold).sum()) - base) * voxel_volume_ml
                        for shift in shifts)
    blur_ml = 0.0
    if scan.blur_sigma_vox is not None:
        blurred = gaussian_filter(scan.image, sigma=scan.blur_sigma_vox)[region]
        blur_ml = (int((blurred >= threshold).sum()) - base) * voxel_volume_ml
    # np.float64 sd: float32 / python float stays float32 under nep 50 and the last digits move
    flip = ndtr(-np.abs(values - threshold) / noise_sd_log)
    noise_var_vox = float((flip * (1 - flip)).sum()) * smoothness(scan.image, nawm, zooms_mm)
    return ScanEffect(contrast_ml, blur_ml, noise_var_vox)


def scanner_range(pair: PairData, domain: np.ndarray, mu: tuple[float, ...]) -> ScannerRange:
    """supplied-mask change with the range a scanner-only change could explain"""
    any_lesion = pair.m1 | pair.m2
    if not any_lesion.any():
        return NO_LESION_RANGE
    shifts = sorted(set(mu))
    blur_ratio = pair.blur_ratio if pair.blur_valid else DEFAULT_BLUR_RATIO
    sigma1, sigma2 = blur_sigmas_vox(blur_ratio, pair.zooms_mm)
    crop = mask_crop(any_lesion, pair.zooms_mm)
    scan1 = CroppedScan(pair.x1[crop].astype(np.float32), pair.m1[crop], sigma1)
    scan2 = CroppedScan(pair.x2[crop].astype(np.float32), pair.m2[crop], sigma2)
    domain_crop = domain[crop]
    noise_sd_log = float(pair.emp_noise) / np.sqrt(2.0)
    effect1 = scan_effect(scan1, domain_crop, pair.zooms_mm, shifts, noise_sd_log)
    effect2 = scan_effect(scan2, domain_crop, pair.zooms_mm, [-shift for shift in shifts], noise_sd_log)

    blur_ml = 0.0
    if sigma1 is not None:
        blur_ml = effect1.blur_ml
    elif sigma2 is not None:
        blur_ml = -effect2.blur_ml
    # scan 1 sees +mu, scan 2 sees -mu, both estimate the same scanner gain in v2 - v1
    contrast_pairs = zip(effect1.contrast_ml, effect2.contrast_ml, strict=True)
    scanner = sorted(0.5 * (c1 - c2) + blur_ml for c1, c2 in contrast_pairs)
    noise_sd_ml = float(np.sqrt(effect1.noise_var_vox + effect2.noise_var_vox) * voxel_ml(pair.zooms_mm))
    dv_ml = mask_change_ml(pair)
    lo_ml = dv_ml - max(0.0, scanner[-1]) - Z_95 * noise_sd_ml
    hi_ml = dv_ml - min(0.0, scanner[0]) + Z_95 * noise_sd_ml
    return ScannerRange(
        dv_ml=dv_ml,
        lo_ml=float(lo_ml),
        hi_ml=float(hi_ml),
        scanner_ml=(scanner[0], scanner[-1]),
        contrast_ml=(scanner[0] - blur_ml, scanner[-1] - blur_ml),
        blur_ml=blur_ml,
        noise_sd_ml=noise_sd_ml,
    )
