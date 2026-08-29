## Mapillary / MTSD

### 1. Classifier Training

- `train_mobilenetv3_mapillary_baseline.py`
- `train_convnext_mapillary_baseline.py`
- `train_mapillary_efficient.py`

### 2. Attack Generation

- `generate-attacks-unified.py`
- `generate_env_unified.py`

### 3. Detection

Run the detection-related files in the following order:

1. `eval_clean_unified.py`
2. `eval_attacked_unified.py`
3. `compute_thresholds_unified.py`
4. `extract_js_consistency_mapillary.py`
5. `analyze_js_consistency_mapillary.py`
6. `extract_deep_features_mapillary.py`
7. `analyze_mahalanobis_mapillary.py`
8. `detect_unified.py`
9. `detect_mapillary_anomalies.py`
10. `combine_detector_js_mapillary.py`

### 4. Reconstruction

- `train_mapillary_ddpm.py`
- `extract_for_ddpm_unified.py`
- `extract_for_ddpm_combined_js_mapillary.py`
- `reconstruct_ddpm_mapillary_combined_js_final.py`

### 5. Final Evaluation

- `recompute_mtsd_msp.py`
- `runtime_from_previous_results.py`

---

## GTSRB

### 1. Classifier Training

- `train_gtsrb_three_models.py`

### 2. Attack Generation

- `generate_gtsrb_all_attacks_three_models.py`

### 3. Detection

Run the detection-related files in the following order:

1. `extract_gtsrb_all_signals.py`
2. `fresh_gtsrb_detection.py`
3. `eval_png_folder_js.py`
4. `calibrate_js_gtsrb.py`
5. `evaluate_gtsrb_ours_js.py`
6. `compare_ours_js_gtsrb.py`
7. `run_gtsrb_ours_js_three_models.py`

### 4. Reconstruction

- `train_ddpm_gtsrb.py`
- `extract_gtsrb_ours_js_suspicious.py`
- `reconstruct_ddpm_gtsrb.py`

### 5. Final Evaluation

- `recompute_gtsrb_msp.py`

---

## Physical Attacks

### 1. Dataset

- `Clean Dataset/`
- `attacked/`

### 2. Detection

Run the physical detection files in the following order:

1. `01_eval_physical_signals.py`
2. `02_compute_physical_thresholds.py`
3. `03_detect_physical_old.py`
4. `04_extract_physical_js.py`
5. `05_analyze_physical_js.py`
6. `06_combine_old_js_physical.py`

### 3. Reconstruction

- `07_extract_physical_ddpm_input.py`
- `08_reconstruct_physical_ddpm.py`

### 4. Additional Physical Analysis

- `09_physical_recalibration_full_analysis.py`
- `10_ddpm_all70_ablation.py`
- `evaluate_physical_qr.py`