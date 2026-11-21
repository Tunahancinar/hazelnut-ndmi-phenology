# hazelnut-ndmi-phenology

Data and code for the article "Phenology and validation design shape machine learning attribution of a Sentinel-2 canopy moisture index in forest adjacent hazelnut stands of Türkiye".

Author: Tunahan Çınar, Department of Forest Engineering, Düzce University (tunahancinar@duzce.edu.tr, ORCID 0000-0002-6473-0884)

## Contents

### data

`parcel_year_table/parcel_year_YYYY.csv` holds one row per parcel and year after quality control, split into one file per year (79,728 rows in total, 10,066 pure hazelnut stands, 2019 to 2026).

- PID: anonymous parcel identifier, shared by all tables
- YEAR: year
- RD: regional directorate of the General Directorate of Forestry
- BLOCK_10KM: code of the 10 km block used for spatial cross validation
- DNDMI: NDMI of July to August minus NDMI of April to May, the response
- SDI: standardized decline index, ΔNDMI standardized within each parcel
- all other columns are the 29 predictors retained after the correlation and VIF screening, coded as in Table 2 of the article

Parcel geometries and coordinates are not included, because the stand boundaries belong to the General Directorate of Forestry.

Other tables (NDMI values multiplied by 10,000, YIL = year):

- `parcel_spring_reference_all_stands.csv`: parcel mean NDMI at the spring NDVI peak (P) for all stands and years, and the May to June median (M) where available
- `parcel_spring_reference_subsample.csv`: parcel mean M and P for the random 30% subsample
- `parcel_spring_peak_doy_subsample.csv`: day of year of the spring NDVI peak observation (parcel median) for the subsample
- `mayjune_centroid_ndmi.csv`: May to June (M) and July to August (J) NDMI at the central pixel of each parcel
- `spring_reference_centroid_ndmi.csv`: April to May median (A) and spring NDVI peak (P) NDMI at the central pixel of each parcel
- `tuik_province_hazelnut_2016_2025.csv`: province hazelnut orchard area (A, decare), production (P, tonne) and yield (kg per decare) from the Turkish Statistical Institute

### scripts

Google Earth Engine

- `gee_parcel_extraction.js`: Sentinel-2, ERA5-Land, MODIS, terrain and soil variables for each parcel and year
- `gee_mayjune_reference.js`, `gee_spring_peak_reference.js`: spring reference NDMI at parcel centroids
- `gee_parcel_ref.js`: parcel mean May to June and spring NDVI peak NDMI
- `gee_peak_doy.js`: day of year of the spring NDVI peak observation

Python

- `03_model_findik.py`: data preparation, quality control, the ten algorithms and spatial and temporal validation
- `12_collinearity.py`: correlation, VIF and Moran's I analysis
- `06_post.py`: SHAP interpretation, random versus spatial cross validation, 2025 bias test
- `11_model_maps.py`, `14_fig_review.py`, `28_fig_doy.py`, `10_workflow_pptx.py`: figures
- `13_review_checks.py`: baselines, block bootstrap, block size, cell level check, imputation, leave one year out validation and SDI sensitivity
- `15_tuning.py`: nested spatial cross validation of hyperparameters
- `16_yield_check.py`: comparison with province yields
- `19_ref_compare.py`, `22_ref_compare_parcel.py`, `23_peak_sensitivity.py`: spring reference comparisons on centroids and parcel means
- `20_nested_selection.py`: correlation and VIF screening repeated inside each training fold
- `21_rd_eval.py`, `26_loro.py`: evaluation by regional directorate and leave one regional directorate out validation
- `24_climate_robust.py`: climate correlations with and without 2025
- `25_shap_bootstrap.py`: normalized SHAP importance of elevation with a 10 km block bootstrap
- `27_peak_doy.py`, `29_elev_shap_vs_doy.py`: spring peak day of year by elevation
- `30_full_ref.py`, `31_combine_ref.py`: all stands analysed with the April to May and the spring NDVI peak references

The scripts were run with Python 3.11, scikit-learn, XGBoost, LightGBM, CatBoost and shap. They were written for the original working environment, so input and output paths are given as command line arguments or as absolute paths at the top of each script and must be adapted. Some comments and variable names are in Turkish.

## Licence

Code: MIT licence (see LICENSE). Data tables: Creative Commons Attribution 4.0 International (CC BY 4.0).

## Citation

Please cite the article and this repository: https://doi.org/10.5281/zenodo.24423674
