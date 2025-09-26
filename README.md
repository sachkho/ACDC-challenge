# ACDC Cardiac Disease Classification — Random‑Forest Pipeline

This repo contains a compact, **interpretable** supervised‑learning pipeline to classify cardiac pathologies from cine‑MRI using anatomical features and a Random Forest classifier. It targets the ACDC challenge setup with five diagnostic classes: **Healthy controls, Myocardial infarction, Dilated cardiomyopathy, Hypertrophic cardiomyopathy,** and **Abnormal right ventricle** (ED/ES MRI volumes per subject).

## Why this approach?
- **Interpretability first**: feature engineering on clinically meaningful volumes/ratios (LV/RV volumes at ED/ES, myocardium volume, ejection fractions, RV/LV and Myo/LV ratios) rather than black‑box end‑to‑end CNNs.
- **Small‑data‑friendly**: tree ensembles work well with tabular, low‑sample settings and give straightforward feature importances.
- **Practical**: robust to missing LV masks thanks to a lightweight **slice‑wise LV reconstruction** inside the myocardium (hole‑filling), enabling a consistent feature set on train and test.

> The overall setup follows the ACDC challenge protocol with ED/ES volumes and partial segmentations; we reconstruct a missing LV mask when needed and train on feature tables derived from the segmentations. See the short report in `challenge_ima205 (2) (1).pdf` for the full narrative and rationale.  

## Repository contents
- `notebook_propre.ipynb` — full pipeline: data I/O, LV reconstruction, feature engineering, model tuning, CV evaluation, and test inference.
- `challenge_ima205 (2) (1).pdf` — 5‑page write‑up of the method, features, CV setup, and discussion.
- (generated) `submission_finalfinal.csv` — submission file produced by the notebook.

## Method overview
1. **Data & masks**  
   Two time points per subject: **End‑Diastole (ED)** and **End‑Systole (ES)**; segmentation labels: RV=1, Myocardium=2, LV=3. Missing LV masks are reconstructed slice‑wise by hole‑filling within the myocardium mask to recover a plausible LV cavity.

2. **Feature extraction (per subject & time point)**  
   - Volumes (liters) from voxel counts × voxel volume (NIfTI header).  
   - **Ejection fractions**: EF = (VED − VES) / VED.  
   - Ratios: **RV/LV**, **Myo/LV** at ED & ES.  
   - Anthropometrics from metadata (height/weight), enabling indexing if needed.  
   - Extra engineered signals: ΔLV/ΔRV volumes (ED−ES), mean EF, total ED volume.

3. **Modeling**  
   - **RandomForestClassifier** (`class_weight='balanced'`).  
   - Hyper‑parameter search with **RandomizedSearchCV** over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`.  
   - **5‑fold Stratified CV** with per‑class **precision/recall/F1** and global accuracy.  
   - Final model refit on full train, then inference on test to build the submission CSV.

## Quickstart
> Requires Python 3.10+

```bash
# 1) Create env (optional)
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 2) Install deps
pip install -U numpy pandas scikit-learn matplotlib nibabel scikit-image scipy opencv-python ipykernel

# 3) Open the notebook
jupyter notebook notebook_propre.ipynb
```

**Data layout expectation (example):**
```
data/
  train/
    images/ED/*.nii.gz
    images/ES/*.nii.gz
    masks/ED/*.nii.gz
    masks/ES/*.nii.gz
  test/
    images/ED/*.nii.gz
    images/ES/*.nii.gz
    masks/ED/*.nii.gz        # LV may be missing → reconstructed
    masks/ES/*.nii.gz
metadata/
  train_metadata.csv
  test_metadata.csv
```

Update the paths in the first cells of the notebook to point to your local folders.

## Reproducing results
1. Run the **pre‑processing + LV reconstruction** cells.  
2. Run **feature extraction** to build `train_features_df` / `test_features_df`.  
3. Execute **hyper‑parameter search**; the notebook prints the best CV accuracy and params.  
4. Run the **per‑class metrics plots** and the **final fit**.  
5. Produce `submission_finalfinal.csv` with the last cell (saved in the repo root by default).

## Notes on performance
- The approach achieves strong cross‑validated accuracy on this small dataset while remaining fully interpretable. See the PDF report for the cross‑validation summary and class‑wise behavior.  
- Given the limited sample size, performance is sensitive to feature quality and class balance; the LV reconstruction and engineered ratios help stabilize generalization.

## Citing / References
- Project report: see `challenge_ima205 (2) (1).pdf` for method justification, metrics, and discussion.
- Related paper mentioned in the report: *Automatic Cardiac Disease Assessment on cine‑MRI via Time‑Series Segmentation and Domain‑Specific Features* (Springer, 2017).

## License
This project is provided for academic purposes; check dataset licenses/terms before redistribution.
