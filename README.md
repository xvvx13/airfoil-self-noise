# Predicting Airfoil Self-Noise across Wind Speeds and Angles of Attack

This aerospace engineering mini-project predicts the scaled sound pressure level of a NACA 0012 airfoil at a specified frequency. The test withholds entire speed–angle combinations. It measures both level error in decibels and how closely the predicted loudest band matches the measured one.

## Reproduce the study

Use Python 3.12 (tested on 3.12.14). From the directory containing this README, run the following commands. On Windows, activate the virtual environment with `.venv\Scripts\activate` in place of the second command.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python study.py train
python study.py predict examples/spectrum.csv --output example_predictions.csv
python plot_spectra.py
```

Training recreates the three model comparisons, all result tables and the selected fitted model. Its last step prints the validation and test metrics. The included fitted model also supports immediate inference after installing the pinned dependencies. The example contains 15 frequencies at 55.5 m/s, zero angle of attack and 0.3048 m chord from the held-out test. At 800 Hz the expected random-forest output is approximately **128.23 dB**. The measured level is 128.086 dB. To predict new inputs, supply a CSV with the same five column names as the example. A spectrum requires one row per frequency.

## Data and units

The included `data/airfoil.csv` contains UCI's six original columns and all 1,503 measurements in source order. The first five columns are the prediction inputs. There are no missing values or duplicate five-input rows. No labels were generated for this project.

| CSV column | Quantity | Unit |
| --- | --- | --- |
| frequency | Acoustic frequency | Hz |
| attack-angle | Angle of attack | degrees |
| chord-length | Airfoil chord | m |
| free-stream-velocity | Free-stream speed | m/s |
| suction-side-displacement-thickness | Suction-side boundary-layer displacement thickness | m |
| scaled-sound-pressure | Measured scaled sound pressure level (target) | dB |

The original data can be downloaded from the [UCI dataset page](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) or the [UCI CSV endpoint](https://archive.ics.uci.edu/static/public/291/data.csv). The bundled copy preserves numerical content and column order, with LF line endings. The bundled CSV SHA-256 is `5cbab0091754b951805a875e97a813439b16e033c2c86b40e04d772b2426c6cb`.

## Experiment

1. Group the data by the exact recorded pair of free-stream speed and angle of attack. A speed–angle pair can contain multiple chords. A spectrum contains all sampled frequencies at a fixed speed, angle, chord and displacement thickness.
2. Reserve 20% of groups using one `GroupShuffleSplit` with seed 7001. There are 67 development groups (1,192 rows, 84 spectra) and 17 test groups (311 rows, 22 spectra). The five `GroupKFold` development folds also keep conditions together.
3. Fit raw ordinary least squares (OLS), curved-spectrum OLS and a random forest with fixed settings. The curved model uses log base 2 of frequency in kHz, its square, angle, and log base 2 of speed, chord and thickness, plus interactions between log frequency and each of the four other variables. The forest uses raw inputs, 300 bootstrap trees, all features at each split, at least two samples per leaf, one worker and seed 7001. All models start from scratch.
4. Choose the model with the lowest pooled out-of-fold MAE. Refit each model on all development rows and evaluate its predictions on the held-out test rows. The saved estimator is the validation winner. MAE and RMSE are row-weighted and measured in dB. R² measures explained variation relative to the test mean.
5. For each test spectrum, sort measured frequencies from low to high. Record the sampled band with the highest level in the measured and predicted curves. An exact match means the same band. Within one band includes an exact match or an adjacent sampled frequency. Ties go to the lowest maximizing frequency for all models and measurements. Two measured test spectra had tied maxima. Each spectrum contributes one vote.

| Model | Validation MAE (dB) | Test MAE (dB) | Test RMSE (dB) | Test R² | Exact peak | Within one band |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw OLS | 3.88 | 3.65 | 4.70 | 0.435 | 0/22 | 1/22 |
| Curved-spectrum OLS | 3.03 | 2.77 | 3.67 | 0.655 | 5/22 | 16/22 |
| Random forest | 2.13 | 1.62 | 2.25 | 0.871 | 10/22 | 19/22 |

The results apply to sampled bands in this wind-tunnel dataset. All four distinct wind speeds are represented in development. Transfer to an entirely new wind speed, airfoil family, or microphone geometry requires a separate test. Inference also assumes the suction-side displacement thickness is available.

## Project files

| Path | Contents |
| --- | --- |
| `study.py` | Training, grouped validation, held-out tests, peak scoring and inference |
| `plot_spectra.py` | Rebuilds the report figure from the saved test predictions |
| `results/splits.csv` | Zero-based UCI CSV record number, speed, angle, split and validation fold |
| `results/metrics.json` | Full-precision metrics, split counts, model selection and peak counts |
| `results/development_oof_predictions.csv` | One out-of-fold prediction per model and development row |
| `results/test_predictions.csv` | Measured inputs, targets, and predictions for the held-out rows |
| `results/test_peaks.csv` | Exact peak frequencies, band error and level MAE by spectrum and model |
| `results/fitted.joblib` | Selected fitted random forest |
The two-page report is submitted separately.

## Report and authorship

The PDF has editable fields for the project author(s), optional student ID(s) and GitHub URL. For a typeset version, set `\AuthorNames`, `\StudentIDs` and `\RepositoryURL` near the beginning of `report/report.tex`. From the repository root, compile with a standard TeX Live installation:

```sh
cd report
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

The researchers listed below created the dataset. The project byline is reserved for the student's or team's actual submission names. OpenAI ChatGPT (Codex) generated the analysis code, ran the experiments, made the figure and drafted the written materials.

## Credits

- T. Brooks, D. Pope and M. Marcolini. *Airfoil Self-Noise* (1989), UCI Machine Learning Repository. The dataset is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). [Dataset DOI](https://doi.org/10.24432/C5VW2C)
- T. F. Brooks, D. S. Pope and M. A. Marcolini. *Airfoil Self-Noise and Prediction*. NASA Reference Publication 1218 (1989). [NASA report](https://ntrs.nasa.gov/citations/19890016302)
- F. Pedregosa et al. *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research 12 (2011), 2825–2830. The models and grouped splitting procedures use scikit-learn. [Software citation](https://jmlr.org/papers/v12/pedregosa11a.html)
