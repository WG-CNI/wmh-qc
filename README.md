# wmh-qc

Quality control for longitudinal white matter hyperintensity (WMH) volume change on FLAIR MRI when the scans come from different scanners or protocols. The masks come from any segmentation method; the QC assesses them, it does not segment.

For every scan pair it reports:

- **dV**: change of the supplied mask volume, later minus earlier. Never replaced by an own estimate.
- **Scanner-effect range** around dV: predicted contrast change (from TE, TR, TI, field), blur difference and patient noise. Scanner effects only, not a confidence interval.
- **Transition class** of the scanner change, from the acquisition metadata.
- **Flags** from an image remeasurement: `bidirectional_growth_and_shrink`, `coherent_dimming_blocks_growth`, `contradiction`.
- **Verdict**: `increase` / `decrease` when no flag is raised and the range excludes zero, otherwise `not_resolved`. `no_call` when TE, TR, TI or field strength are missing.

## Install

```
pip install .            # qc
pip install ".[prepare]" # plus SimpleITK for `wmh-qc prepare`
pip install ".[test]"    # then: pytest
```

Python 3.11+, NumPy 2. `prepare` also needs FreeSurfer 7.4 (`mri_synthstrip`, `mri_synthseg`) and FSL `flirt` on `PATH`. On recent Fedora/RHEL, FreeSurfer needs `libxcrypt-compat`.

## Usage

**1. Prepare**: raw FLAIR plus its WMH mask per scan, in acquisition order. Skull stripping, rigid registration to the first scan, N4, SynthSeg. About 15 min per scan.

```
wmh-qc prepare --scan baseline.nii.gz baseline_mask.nii.gz \
               --scan followup.nii.gz followup_mask.nii.gz \
               --out prepared/
```

`--out` must be a new folder. Images prepared elsewhere can be used in the same layout: `prepared/scanN/flair_n4.nii.gz`, `mask.nii.gz`, `synthseg.nii.gz` and optionally `synthseg_unfilled.nii.gz`, all on the first scan's grid.

**2. Run**: one `--meta` per scan, in the same order. Each is the FLAIR's DICOM folder or a JSON sidecar.

```
wmh-qc run --prepared prepared/ --meta dicom/baseline/ --meta dicom/followup/ --out qc.json
```

Sidecars are dcm2niix/BIDS (times in s) or this format (times in ms):

```json
{"te": 82, "tr": 8500, "ti": 2440, "field": 1.5, "vendor": "SIEMENS",
 "model": "SymphonyTim", "slice_thickness": 3.0, "date": "2021-03-04T10:15:30"}
```

With more than two scans, every consecutive pair and every pair with the baseline is reported.

## Output

`qc.json` holds one record per pair:

| field | meaning |
|---|---|
| `earlier`, `later`, `pair` | scan indices, `consecutive` or `baseline` |
| `dv_ml`, `range_lo_ml`, `range_hi_ml` | mask change and scanner-effect range |
| `scanner_ml`, `contrast_ml`, `blur_ml`, `noise_sd_ml`, `mu` | range components |
| `transition_class`, `severity`, `changed_fields`, `same_scanner` | scanner transition |
| `remeasured_net_ml`, `growth_ml`, `shrink_ml`, `flags`, `verdict` | remeasurement and result |

## Notes

- The run stops with a message on bad input: non-binary or wrong-scan masks, normalised FLAIRs, non-SynthSeg label maps, images off the baseline grid. Messages name files, never folders.
- Size thresholds are in mm and mm³, calibrated on the paper's 0.5 mm grid.
- On the paper's data the package reproduces the published results. `prepare` is for new data and differs from the paper's preprocessing.
