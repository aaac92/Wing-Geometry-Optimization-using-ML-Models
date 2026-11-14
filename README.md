# Wing Geometry Optimization using ML Models

Machine-learning surrogates that search for wing configurations/shapes that maximize the aerodynamic efficiency (E = CL/CD) of a finite wing. They aim to beat the best design of a simulated dataset while running far fewer simulations than a blind search.

Each design is labeled by a simulation chain:

```
'n' CST coefficients ─► airfoil (.dat) ─► XFOIL polar ─► key points (GPR fit) ─► AVL wing configuration/s ─► CL, CD, E
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
   - `FoilGeoFeatures.get_minimum_camber()`: the counterpart of `get_maximum_camber` used for negative camber, since `get_minimum_camber` reads as 0 when the camber becomes negative.
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
   - To reproduce dataset 17 and 17.5 use [RobotLocomotion](https://github.com/RobotLocomotion)'s CMake compilation in Linux. This will compile older versions of both XFOIL (6.97) and AVL (3.32). Make sure double precision is on. AVL 3.32 takes `CDCL` arguments per section and cannot take them per surface, while AVL 3.40 can use both notations.
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

Reusable modules use command line functionality for self-tests under `if __name__ == "__main__"`.

## Creating a study

### 1. Choose a case

A case needs three things:
- a reference wing, ideally with experimental data to avoid unrealistic results during simulation;
- a fixed flight condition (Re, Mach);
- an objective that the AVL output can compute.

This work replicates the wind-tunnel wing of Hossain et al. (2011)[^hossain]: chord 0.121 m, Re = 2.5×10⁵, Mach = 0.0928. It uses three configurations: `rec` (no winglet), `flat` (flat winglet) and `60DEG` (60° winglet). Before building a dataset, check that the base airfoil reproduces the reference results. In this case results cannot be replicated since AVL's Vortex Lattice method cannot represent correctly how a winglet performs on a wing. Using AVL the results where opposite of what is expected of a wing using a winglet, and the rectangular configuration outperformed the other two in almost all airfoil shapes.

### 2. Describe the wing in AVL

Add a class to `simulation_support/geo_file_creation.py`, using `HossainWingRec_Creation` as the template:
- `__init__(key_points_data, airfoil_abs_paths, naca_key_points_data, output_dir, verbose, pb)`
- `create_geo_files(progress_callback=None)` writes `<airfoil>_<identifier>_geo.avl`. Each section takes the `CDCL` line from the airfoil's key points (CL and CD at minimum CL, minimum CD and maximum CL).
- `save_error_log(path)`

Then:
- export the class in `simulation_support/__init__.py`;
- register its identifier in `GEO_CREATORS_BY_IDENTIFIER` (`simulation_support/airfoil_simulation.py`), which the mending, the sweep check and the inference use to re-simulate;
- put the AVL run case (flight condition) in `data/support/`, like `Hossain_sqwing_case.run`.

In this case the identifier becomes the configuration's one-hot column in the regressor dataset. If using a continuous feature like geometric twist or taper ratio then the column should not be a one-hot column.

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

- **Split by shape.** The test split (20%), the cross-validation folds and the early-stopping slice are all split by airfoil shape (`ai_support/data_split.py`). A split by row could test the models on shapes they have already seen.
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
- **Test split.** The fine-tuning never uses the dataset's test shapes (`--test-split-seed`, default 42, the models' training seed). The final models are scored on them, next to the score of the original models: a lower MAE means the fine-tuning improved them on unseen shapes.
- **Output.** The results go to `results/dataset_<N>/<date>_m_seed<seed>/`. The run also writes the expanded dataset (`dataset_<N>.<i>`) and the fine-tuned models (`<Name>_<version>.<i>`), where `i` is the number of iterations.
- **Resume.** Continue a stopped run with `--resume <run folder>`.

## Current dataset and findings

### Dataset 17.5

Dataset 17 was simulated with a sequential alpha sweep that produced a non-physical low-drag branch. Dataset 17.5 rebuilds it with the `zero_out` sweep:
- **Shapes:** 147,456 CST combinations on a NACA 0012 base. 134,765 pass the geometric filters.
- **Regressor rows:** 404,031 (134,677 shapes × 3 configurations). 85 shapes failed in simulation, and 3 were dropped because their polar is below the laminar CD limit.
- **Mending:** flagged 1,701 rows, and re-simulation confirmed all of them.
- **Dataset best:** E = 26.64.

Test split by shape (seed 42): 80,808 rows from 26,936 shapes the models never saw.

| Model | Type | Test MAE of E | R² | Max error |
| --- | --- | --- | --- | --- |
| Odysseus_1.5 | MLP, `raw` | 0.293 | 0.9815 | 11.9 |
| Murata_1.5 | MLP, `sum_station` | 0.314 | 0.9812 | 11.3 |
| Marbas | LightGBM, `sum_station` | 0.328 | 0.9798 | 12.1 |
| Agares_1.5 | LightGBM, `sum_station` | 0.370 | 0.9774 | 11.8 |

The earlier split by row put the other configurations of a test shape in the training rows, which gave optimistic scores (MAE 0.13–0.16). The classifier Calipso_1.5 learns the geometric filters; it classified all 29,492 test shapes correctly.

### Iterative inference

Three runs with the models retrained with the split by shape, on the same dataset, with seeds `e021ce33`, `67` and `69` (`results/dataset_17.5/20261111_m_seed*`). Each took 8–9 iterations and 1.4–2.0 h. Every lane beat the dataset best (target E = 26.66, re-simulated) in every run.

| Lane | Verdicts (e021ce33 / 67 / 69) | Best E, mean ± SD | vs target | Shapes simulated to beat the target |
| --- | --- | --- | --- | --- |
| Marbas (LightGBM) | !! / !! / !! | 29.54 ± 0.31 | +10.8% | 7 / 7 / 6 |
| Odysseus_1.5 (MLP) | ?? / ?? / !! | 29.19 ± 0.26 | +9.5% | 23 / 35 / 25 |
| Murata_1.5 (MLP) | !! / ?? / ?? | 28.50 ± 0.41 | +6.9% | 30 / 30 / 30 |
| consensus | ?? / !! / ?? | 29.71 ± 1.03 | +11.4% | 18 / 11 / 7 |
| random (control) | ?? / !! / !! | 28.56 ± 0.77 | +7.1% | 32 / 30 / 34 |

- **Speed.** Marbas beats the dataset best after about 7 simulated shapes in every run; the random control needs about 30.
- **One optimum region.** 10 of the 15 lane winners lie within about 1 grid step of another winner (E 28.2–29.5, rec configuration), outside the dataset window with cst_l_1, cst_l_5 and cst_u_5 above the grid. The next dataset should widen those bounds.
- **Best validated design.** Marbas `s5b1f793f71` (seed 69): E 29.90 (+12.2%), cold-start agrees, filters pass.
- **Murata_1.5.** Its iteration-0 picks are predicted at E ≈ 34 but simulate at ≈ 19 in every run, and its last iteration is wrong the same way. Its lane stops on these failed picks, not at an optimum.
- **Verdicts vary between seeds.** The cold-start check needs ≥ 99% of about 32 pre-stall angles on the same branch, so one angle decides it. The curvature filter rejects 19 of the 45 top-3 designs, including random-lane designs. Compare lanes by best E and shapes simulated, not by the verdict.
- **Fine-tuning.** On the test split, Murata_1.5 improved in every run (MAE 0.314 → 0.274–0.285), Odysseus_1.5 in two (0.294 → 0.260, +0.003 in the third), and Marbas did not change.

## License

Copyright (C) 2026 Andrés A. Alvarado Camacho

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. See [LICENSE](LICENSE).

XFOIL and AVL are separate programs distributed by their authors under the GNU GPL. They are not included in this repository.

[^hossain]: A. Hossain, A. Rahman, J. Hossen, A. K. M. P. Iqbal and M. I. Zahirul, "Prediction of aerodynamic characteristics of an aircraft model with and without winglet using fuzzy logic technique," *Aerospace Science and Technology*, vol. 15, no. 8, pp. 595–605, 2011.
