# Wing Geometry Optimization using ML Models

Machine-learning surrogates that search for airfoil shapes that maximize the aerodynamic efficiency (E = CL/CD) of a finite wing. They aim to beat the best design of a simulated dataset while running far fewer simulations than a blind search.

Each design is labeled by a simulation chain:

```
10 CST coefficients ─► airfoil (.dat) ─► XFOIL polar ─► key points (GPR fit) ─► AVL wing, 3 configurations ─► CL, CD, E
```

- A **classifier** predicts whether a shape passes the geometric filters.
- **Regressors** (MLP, LightGBM) predict E for each wing configuration.
- An **iterative trust-region search** proposes new shapes, simulates them, and fine-tunes the models on the results.

## Repository layout

```
bin/        XFOIL and AVL executables (not distributed, see Installation)
data/
  support/  base airfoil (NACA 0012), NACA 0012 key points, AVL run case
  datasets/ dataset_<N>/regressor_dataset_<N>.csv and classifier_dataset_<N>.csv
  logs/     generation parameters and stage summaries of every dataset
models/     dataset_<N>/<model type>/<Name>_*   (trained models; .joblib files use Git LFS)
results/    dataset_<N>/<date>_m_seed<seed>/    (inference runs)
src/        packages, drivers and utilities
```

## Installation

1. **Python 3.12 or newer.** Install [PyTorch](https://pytorch.org/get-started/locally/) first, choosing the CUDA build for your GPU. Then install the project in editable mode:
   ```bash
   pip install -e .
   ```
2. **Patch `cst_modeling`.** The code expects two additions to `cst_modeling/foil.py` (package `cst_modeling3d` 0.2.8) that the PyPI release lacks:
   - A `verbose=True` argument that silences the error prints of `check_validity`. `FoilGeoFeatures` and `FoilModification` also take it and pass it on.
   - `FoilGeoFeatures.get_minimum_camber()`: the counterpart of `get_maximum_camber` for negative camber.
3. **XFOIL and AVL** (Mark Drela and Harold Youngren, MIT, GPL):
   - Get them from the official pages: [XFOIL](https://web.mit.edu/drela/Public/web/xfoil/) and [AVL](https://web.mit.edu/drela/Public/web/avl/). Download the Windows executables, or build from source on Linux.
   - Place them in `bin/` as `xfoil` and `avl`, with the `.exe` extension on Windows.
   - Check that both work:
     ```bash
     python src/simulation_support/polar_creation.py
     ```
     ```bash
     python src/simulation_support/avl_simulation.py
     ```
   - XFOIL cuts file names at 64 characters and AVL cuts force-file names at 80 (`simulation_support/file_names.py`). The drivers therefore work in short temporary folders (`C:\` on Windows, `~/.<name>` on Linux).
4. **RAR command line** (`Rar.exe` from WinRAR on Windows, `rar` on Linux). The dataset builder uses it to archive each stage.
5. **Git LFS** (`git lfs install`) before cloning, so the LightGBM `.joblib` models download in full.

## Packages (`src/`)

| Package | Purpose | Main contents |
| --- | --- | --- |
| `geometric_support` | Shape generation and geometric filters | `Airfoil_Generator` (CST increments on a base airfoil), `create_modification_combinations` (CST grid), `Validate_Airfoil`, `filter_settings.py` (`DATASET_FILTERS`) |
| `simulation_support` | XFOIL, key points and AVL | `Xfoil_Polar_Creator`, `alpha_sweep` (sweep modes, default `zero_out`), `Key_Point_Extraction`, `geo_file_creation` (AVL wing writers), `AVL_Run_Case`, `simulate_airfoil_once` (whole chain for one shape), `sweep_protocol_check` |
| `file_handling_support` | Datasets and files | `Training_Data_Generator` (training CSVs), `Dataset_Rectifier` (neighbour-consistency mending), `dataset_names` / `model_names` (versioning), `compression_Alma` (RAR) |
| `ai_support` | Models and search | `Classifier`, `MLP_Regressor`, `LGBM_Regressor`, `SKNN_Regressor`, `CSTFeatureEngineer` (`raw` / `station` / `sum_station` features), `data_split` (splits by airfoil shape), `scoring` (test-split scores), `model_loading`, `trust_region` (`Dataset_Peaks`, `Trust_Region`, `Lane_Monitor`) |
| `ux_support` | Long runs | progress windows and console bars, recovery files and atomic writes (resume after a crash or power cut) |

| Driver | Purpose |
| --- | --- |
| `Hossain_DB_Gen.py` | Builds a dataset in six resumable stages: airfoils → polars → AVL geometry → forces → training CSVs → mending |
| `MLP_Classifier_Tuning.py`, `MLP_Regressor_Tuning.py`, `LGBM_Regressor_Tuning.py`, `SKNN_Regressor_Tuning.py` | Optuna hyperparameter search plus a final fit |
| `inference_and_testing.py` | Iterative trust-region inference with simulation validation |
| `gpr_validation_study.py` | Validation study of the key-point GPR fit |
| `utilities/ml_gym.py` | Trains one configuration, or updates and retrains saved models |
| `utilities/model_score.py` | Scores saved models on the test split of their dataset |
| `utilities/dataset_rectification.py` | Diagnoses and rectifies an existing dataset |
| `Database_Generation.py`, `AI_Airfoil_Tests.py` | Legacy drivers (datasets ≤ 16), kept for reference |

Reusable modules keep their former command line as a self-test under `if __name__ == "__main__"`.

## Creating a study

### 1. Choose a case

A case needs three things:
- a reference wing, ideally with experimental data to validate the AVL model against;
- a fixed flight condition (Re, Mach);
- an objective that the AVL output can compute.

This work replicates the wind-tunnel wing of Hossain et al. (2011)[^hossain]: chord 0.121 m, Re = 2.5×10⁵, Mach = 0.0928. It uses three configurations: `rec` (no winglet), `flat` (flat winglet) and `60DEG` (60° winglet). Before building a dataset, check that the base airfoil reproduces the reference results.

### 2. Describe the wing in AVL

Add a class to `simulation_support/geo_file_creation.py`, using `HossainWingRec_Creation` as the template:
- `__init__(key_points_data, airfoil_abs_paths, naca_key_points_data, output_dir, verbose, pb)`
- `create_geo_files(progress_callback=None)` writes `<airfoil>_<identifier>_geo.avl`. Each section takes the `CDCL` line from the airfoil's key points (CL and CD at minimum CL, minimum CD and maximum CL).
- `save_error_log(path)`

Then:
- export the class in `simulation_support/__init__.py`;
- register its identifier in `GEO_CREATORS_BY_IDENTIFIER` (`simulation_support/airfoil_simulation.py`), which the mending, the sweep check and the inference use to re-simulate;
- put the AVL run case (flight condition) in `data/support/`, like `Hossain_sqwing_case.run`.

The identifier becomes the configuration's one-hot column in the regressor dataset.

### 3. Define the shape limits

- **CST grid**, in the driver: `n_cst` and the `(start, end, step)` ranges of the leading-edge, intermediate and trailing-edge coefficients of each surface (`ranges_upper`, `ranges_lower`). The values are increments added to the base airfoil (`data/support/naca0012_unit.csv`), and `create_modification_combinations` builds every combination.
- **Geometric filters**, in `geometric_support/filter_settings.py`: thickness, leading-edge sharpness and curvature continuity, monotonic thickness, and camber.
  - Change a threshold or switch a filter off with `filter_settings(camber={"enabled": False})`.
  - Add a filter with an entry in `DATASET_FILTERS` and a check in `FILTER_CHECKS`.
- Preview the effect before a long run with `utilities/cst_range_test.py` and `utilities/filter_plotter.py`.

### 4. Build the dataset

Copy `Hossain_DB_Gen.py` and edit its `__main__` block:
- `DATASET_NUM`;
- base airfoil, CST ranges and filters;
- `configs` (geometry classes and identifiers);
- `ASEQ`, `Re`, `Mach`;
- the mending rule.

Test the whole pipeline on a few airfoils first:

```bash
python src/Hossain_DB_Gen.py --smoke 40
```

```bash
python src/Hossain_DB_Gen.py
```

If the run is interrupted, run the same command again: it resumes at the stage and file where it stopped. The run produces:
- `data/datasets/dataset_<N>/regressor_dataset_<N>.csv`, with columns `cst_l_1..n`, `cst_u_1..n`, one-hot configurations and `E`;
- `classifier_dataset_<N>.csv`, with the CST columns and `valid`: one row per shape, 1 when it passed the geometric filters;
- the generation log and stage summaries in `data/logs/`.

A decimal number (17.5) marks a rebuild of an existing dataset.

Two checks remove non-physical results from the regressor dataset:
- **Laminar limit.** A polar whose lowest CD is below the laminar flat-plate limit, 2 × 1.328 / √Re, is dropped before AVL.
- **Mending.** Rows far from their grid neighbours are simulated again, and corrected or dropped.

### 5. Train models

Set `dataset_num` (and `FEATURE_MODE` for regressors) at the top of each tuning script, then run it:

```bash
python src/MLP_Regressor_Tuning.py
```

Models are saved as `models/dataset_<N>/<type>/<Name>_*`, with a name drawn from the script's name pool. An interrupted search resumes from its recovery folder.

To train one known configuration, or to update saved models on a new dataset (the update becomes version `Name_1.1`), use `utilities/ml_gym.py`. Its docstring has examples.

- **Split by shape.** The test split (20%), the cross-validation folds and the early-stopping slice are all split by airfoil shape (`ai_support/data_split.py`). The three wing configurations of one shape have almost proportional E values, so a split by row would test the models on shapes they have already seen.
- **Test score.** Every trained model gets its test-split score saved next to it, as `<Name>_test_score.txt` and `.json`. Regressors get MAE and R², overall and per wing configuration. The classifier gets its confusion matrix and the recall of the invalid class.
- **Scoring later.** Score any saved model again, for example with another split seed:

```bash
python src/utilities/model_score.py --dataset 17.5
```

### 6. Test the models

```bash
python src/inference_and_testing.py --classifier Calipso_1.5 --models Marbas Odysseus_1.5 Murata_1.5 --dataset 17.5
```

- **Lanes.** The run starts one lane per regressor, plus a `consensus` lane and a `random` control.
- **Each iteration**, every lane samples its trust regions, simulates the best candidates and fine-tunes its own models.
- **Stop rule.** A lane stops when its best E improves by less than 0.5% for 2 iterations in a row.
- **Final check.** At the end, each lane's top 3 designs are re-simulated with a history-free (cold-start) XFOIL sweep.
- **Verdict.** A lane is a `SUCCESS!!` when one of its top 3 designs beats the dataset best, agrees with its cold-start simulation and passes every geometric filter. It is a `SUCCESS??` when its best design beats the dataset best but none passes all three checks.
- **Test split.** The fine-tuning never uses the dataset's test shapes (`--test-split-seed`, default 42, the models' training seed). The final models are scored on them, next to the score of the original models: the difference is what the fine-tuning made them forget.
- **Output.** The results go to `results/dataset_<N>/<date>_m_seed<seed>/`. The run also writes the expanded dataset (`dataset_<N>.<i>`) and the fine-tuned models (`<Name>_<version>.<i>`), where `i` is the number of iterations.
- **Resume.** Continue a stopped run with `--resume <run folder>`.

## Current dataset and findings

### Dataset 17.5

Dataset 17 was simulated with a sequential alpha sweep that produced a non-physical low-drag branch. Dataset 17.5 rebuilds it with the `zero_out` sweep:
- **Shapes:** 147,456 CST combinations on a NACA 0012 base. 134,765 pass the geometric filters.
- **Regressor rows:** 404,031 (134,677 shapes × 3 configurations). 85 shapes failed in simulation, and 3 were dropped because their polar is below the laminar CD limit.
- **Mending:** flagged 1,701 rows, and re-simulation confirmed all of them.
- **Dataset best:** E = 26.64.

Test-split MAE of E, measured with the earlier split by row. These values are optimistic for unseen shapes; the models are being retrained with the split by shape and the table will be replaced:

| Model | Type | Test MAE of E |
| --- | --- | --- |
| Murata_1.5 | MLP, `sum_station` | 0.131 |
| Agares_1.5 | LightGBM, `sum_station` | 0.135 |
| Odysseus_1.5 | MLP, `raw` | 0.162 |

All three have R² ≥ 0.995. The classifier Calipso_1.5 is also being retrained: its earlier dataset repeated every valid shape three times, so its earlier score (F1 = 0.99999) was not a real test.

### Iterative inference

Run `results/dataset_17.5/20261106_m_seede021ce33`: 6 iterations, 3 h. Every lane converged. The run predates the split by shape and the retraining, and will be repeated with the same seed.

| Lane | Verdict | Best E | vs dataset best | Shapes simulated to beat it | Cold-start E of the best design |
| --- | --- | --- | --- | --- | --- |
| Marbas (LightGBM) | SUCCESS!! | 29.31 | +9.9% | 8 | 29.01 |
| Odysseus_1.5 (MLP) | SUCCESS!! | 29.24 | +9.7% | 19 | 28.58 |
| Murata_1.5 (MLP) | SUCCESS?? | 29.24 | +9.7% | 7 | failed |
| consensus | SUCCESS?? | 29.17 | +9.4% | 6 | 28.69 |
| random (control) | SUCCESS?? | 29.34 | +10.1% | 32 | 23.21 |

- **Speed.** The model lanes beat the dataset best after 6–19 simulated shapes. The random control needed 32.
- **Cold start.** The best designs of Marbas, Odysseus and consensus lose 1.0–2.2% of E in the cold-start check. The random lane's best design loses 21%.
- **Validated designs.** Three designs pass the full validation: Marbas `s1c023d7829` (E 29.27, +9.8%) and `s13496103a7` (E 29.18, +9.5%), and Odysseus `sc1ca28d413` (E 28.96, +8.6%). They give those two lanes their `SUCCESS!!`.
- **Outside the dataset.** The best designs lie 0.6–2 grid steps outside the dataset window, mostly in the trailing-edge coefficients. The next dataset should widen those bounds.

## License

Copyright (C) 2026 Andrés A. Alvarado Camacho

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. See [LICENSE](LICENSE).

XFOIL and AVL are separate programs distributed by their authors under the GNU GPL. They are not included in this repository.

[^hossain]: A. Hossain, A. Rahman, J. Hossen, A. K. M. P. Iqbal and M. I. Zahirul, "Prediction of aerodynamic characteristics of an aircraft model with and without winglet using fuzzy logic technique," *Aerospace Science and Technology*, vol. 15, no. 8, pp. 595–605, 2011.
