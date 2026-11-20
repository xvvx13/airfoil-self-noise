"""Train, evaluate, or use the NACA 0012 airfoil self-noise models."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "airfoil.csv"
FREQ = "frequency"
ANGLE = "attack-angle"
CHORD = "chord-length"
SPEED = "free-stream-velocity"
THICK = "suction-side-displacement-thickness"
LEVEL = "scaled-sound-pressure"
INPUTS = [FREQ, ANGLE, CHORD, SPEED, THICK]
CONDITION = [SPEED, ANGLE]
SPECTRUM = [SPEED, ANGLE, CHORD, THICK]
SEED = 7001


def design(frame, model_name):
    inputs = frame[INPUTS].copy()
    if not np.isfinite(inputs.to_numpy()).all():
        raise ValueError("All input values must be finite numbers.")
    if (inputs[[FREQ, CHORD, SPEED, THICK]] <= 0).any().any():
        raise ValueError("Frequency, chord, speed, and thickness must be positive.")
    if model_name == "quadratic_ols":
        # A quadratic in log frequency can turn over inside the measured band.
        log_f = np.log2(inputs[FREQ] / 1000.0)
        inputs[FREQ] = log_f
        inputs["log_frequency_squared"] = log_f ** 2
        for name in [SPEED, CHORD, THICK]:
            inputs[name] = np.log2(inputs[name])
        for name in [ANGLE, SPEED, CHORD, THICK]:
            inputs["frequency_x_" + name] = log_f * inputs[name]
    return inputs


def metrics(actual, estimate):
    return {
        "mae_db": float(mean_absolute_error(actual, estimate)),
        "rmse_db": float(np.sqrt(mean_squared_error(actual, estimate))),
        "r2": float(r2_score(actual, estimate)),
    }


def peak_scores(frame, model_names):
    rows = []
    for key, curve in frame.groupby(SPECTRUM, sort=True):
        curve = curve.sort_values(FREQ)
        freq = curve[FREQ].to_numpy()
        measured_index = int(np.argmax(curve[LEVEL].to_numpy()))
        for model in model_names:
            fitted = curve[model].to_numpy()
            peak_index = int(np.argmax(fitted))
            rows.append({
                **dict(zip(SPECTRUM, key)), "model": model, "n_bands": len(curve),
                "measured_peak_hz": freq[measured_index], "predicted_peak_hz": freq[peak_index],
                "band_error": abs(measured_index - peak_index),
                "tied_predicted_bands": int((fitted == fitted.max()).sum()),
                "tied_measured_bands": int((curve[LEVEL] == curve[LEVEL].max()).sum()),
                "measured_peak_at_endpoint": measured_index in (0, len(curve)-1),
                "spectrum_mae_db": mean_absolute_error(curve[LEVEL], fitted)
            })
    return pd.DataFrame(rows)


def train():
    RESULTS.mkdir(exist_ok=True)
    frame = pd.read_csv(DATA)
    if frame[INPUTS + [LEVEL]].isna().any().any() or frame.duplicated(INPUTS).any():
        raise ValueError("Expected one complete measurement per input combination.")
    group_id = frame.groupby(CONDITION, sort=True).ngroup()
    split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    dev, test = next(split.split(frame, groups=group_id))
    assert set(group_id.iloc[dev]).isdisjoint(group_id.iloc[test])
    folds = list(GroupKFold(n_splits=5).split(dev, groups=group_id.iloc[dev]))
    split_log = frame[CONDITION].copy()
    split_log["split"] = "test"
    split_log["validation_fold"] = pd.array([pd.NA] * len(frame), dtype="Int64")
    split_log.loc[dev, "split"] = "development"
    for number, (fit, valid) in enumerate(folds, start=1):
        assert set(group_id.iloc[dev[fit]]).isdisjoint(group_id.iloc[dev[valid]])
        split_log.loc[dev[valid], "validation_fold"] = number
    split_log.index.name = "row_id"
    split_log.to_csv(RESULTS / "splits.csv")

    models = {
        "raw_ols": LinearRegression(),
        "quadratic_ols": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                max_features=1.0, random_state=SEED, n_jobs=1),
    }
    validation = frame.iloc[dev].copy()
    held = frame.iloc[test].copy()
    validation_metrics = {}
    for name, estimator in models.items():
        x = design(validation, name)
        estimate = cross_val_predict(estimator, x, validation[LEVEL], cv=folds)
        validation[name] = estimate
        validation_metrics[name] = metrics(validation[LEVEL], estimate)
    best = min(models, key=lambda name: validation_metrics[name]["mae_db"])

    test_metrics = {}
    for name, estimator in models.items():
        fitted = clone(estimator).fit(design(validation, name), validation[LEVEL])
        held[name] = fitted.predict(design(held, name))
        test_metrics[name] = metrics(held[LEVEL], held[name])
        if name == best:
            joblib.dump({"name": name, "model": fitted}, RESULTS / "fitted.joblib", compress=3)
    for label, part in [("development_oof", validation), ("test", held)]:
        part.index.name = "row_id"
        part.to_csv(RESULTS / (label + "_predictions.csv"))
    peaks = peak_scores(held, models)
    peaks.to_csv(RESULTS / "test_peaks.csv", index=False)
    counts = {
        key: {"rows": len(part), "conditions": len(part[CONDITION].drop_duplicates()),
              "spectra": len(part[SPECTRUM].drop_duplicates())}
        for key, part in [("development", validation), ("test", held)]
    }
    summary = {"seed": SEED, "selected": best, "counts": counts,
               "validation": validation_metrics, "test": test_metrics, "peaks": {}}
    for name, part in peaks.groupby("model"):
        summary["peaks"][name] = {"exact": int((part.band_error == 0).sum()),
                                  "within_one": int((part.band_error <= 1).sum()),
                                  "total": len(part)}
    (RESULTS / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def predict(path, output):
    bundle_path = RESULTS / "fitted.joblib"
    if not bundle_path.exists():
        raise SystemExit("Train the project first with: python study.py train")
    bundle = joblib.load(bundle_path)
    frame = pd.read_csv(path)
    missing = set(INPUTS) - set(frame)
    if missing:
        raise ValueError("Missing input columns: " + ", ".join(sorted(missing)))
    frame["predicted_scaled_spl_db"] = bundle["model"].predict(design(frame, bundle["name"]))
    if output:
        frame.to_csv(output, index=False)
    else:
        print(frame.to_csv(index=False), end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train", help="Run grouped validation, test scoring, and model fitting.")
    p = commands.add_parser("predict", help="Predict scaled levels for a CSV of inputs.")
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        train()
    else:
        predict(args.input, args.output)
