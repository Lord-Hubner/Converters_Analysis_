#!/usr/bin/env python3
"""Compare traditional ML models on the differential-expression datasets.

The script evaluates every model with the same stratified cross-validation
splits and writes one out-of-fold classification for every input row. Logistic
regression is used as the linear model because the target is categorical.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


RANDOM_STATE = 42
LABEL_COLUMN = "label"
CLASS_NAMES = {
    0: "Downregulated",
    1: "Upgregulated",
    2: "NoDifference",
}
DEFAULT_INPUTS = [
    Path("LUAD_features_nogene.csv"),
    Path("KIRC_features_nogene.csv"),
    Path("PRAD_features_nogene.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate five traditional ML classifiers and classify every row "
            "as Upregulated, Downregulated, or NoDifference."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=DEFAULT_INPUTS,
        help="Input CSV files (default: LUAD, KIRC, and PRAD *_features_nogene.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("traditional_ml_results"),
        help="Directory for metrics, predictions, and final fitted models.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of stratified cross-validation folds (default: 5).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "logistic_regression",
            "xgboost",
            "random_forest",
            "support_vector_machine",
            "naive_bayes",
        ],
        help="Optional subset of models to run.",
    )
    parser.add_argument(
        "--save-final-models",
        action="store_true",
        help="Do not fit and save each model on the complete dataset after evaluation.",
    )
    return parser.parse_args()


def build_models() -> dict[str, object]:
    return {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        max_iter=2_000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        objective="multi:softprob",
                        num_class=len(CLASS_NAMES),
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=5,
                        min_child_weight=2,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        eval_metric="mlogloss",
                        tree_method="hist",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "support_vector_machine": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        C=2.0,
                        kernel="rbf",
                        gamma="scale",
                        probability=True,
                        cache_size=2_000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "naive_bayes": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    GaussianNB(var_smoothing=1e-9),
                ),
            ]
        ),
    }


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Index]:
    frame = pd.read_csv(path, index_col=0)

    features = frame.drop(columns=LABEL_COLUMN)
    labels = pd.to_numeric(frame[LABEL_COLUMN], errors="raise").astype(int)

    non_numeric = features.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric:
        raise ValueError(f"{path} contains non-numeric features: {non_numeric}")

    return features, labels, frame.index


def fit_model(model: object, X: pd.DataFrame, y: pd.Series) -> object:
    fitted = clone(model)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y)

    fitted.fit(X, y, model__sample_weight=sample_weight)
    return fitted


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        #"balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def evaluate_model(
    model_name: str,
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    row_ids: pd.Index,
    splits: list[tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
) -> tuple[dict[str, object], pd.DataFrame]:
    labels = np.array(sorted(CLASS_NAMES))
    predictions = np.empty(len(y), dtype=int)
    probabilities = np.zeros((len(y), len(labels)), dtype=float)
    fold_numbers = np.empty(len(y), dtype=int)
    fold_rows: list[dict[str, object]] = []
    started = perf_counter()

    print(f"  {model_name}:")
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        fold_started = perf_counter()
        fitted = fit_model(model, X.iloc[train_idx], y.iloc[train_idx])
        fold_predictions = fitted.predict(X.iloc[test_idx])
        fold_probabilities = fitted.predict_proba(X.iloc[test_idx])

        predictions[test_idx] = fold_predictions
        probabilities[test_idx] = fold_probabilities
        fold_numbers[test_idx] = fold

        fold_metrics = calculate_metrics(y.iloc[test_idx].to_numpy(), fold_predictions)
        fold_rows.append(
            {
                "model": model_name,
                "fold": fold,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                **fold_metrics,
                "seconds": perf_counter() - fold_started,
            }
        )
        print(
            f"    fold {fold}: macro-F1={fold_metrics['f1_macro']:.4f}"
            #f"balanced accuracy={fold_metrics['balanced_accuracy']:.4f}"
        )

    overall = calculate_metrics(y.to_numpy(), predictions)
    summary = {
        "model": model_name,
        **overall,
        "seconds": perf_counter() - started,
    }

    prediction_frame = pd.DataFrame(
        {
            "row_id": row_ids,
            "fold": fold_numbers,
            "true_label": y.to_numpy(),
            "true_class": y.map(CLASS_NAMES).to_numpy(),
            "predicted_label": predictions,
            "predicted_class": pd.Series(predictions).map(CLASS_NAMES).to_numpy(),
            **{
                f"probability_{CLASS_NAMES[label]}": probabilities[:, position]
                for position, label in enumerate(labels)
            },
        }
    )
    prediction_frame.to_csv(output_dir / f"{model_name}_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / f"{model_name}_fold_metrics.csv", index=False)

    matrix = confusion_matrix(y, predictions, labels=labels)
    pd.DataFrame(
        matrix,
        index=[f"actual_{CLASS_NAMES[label]}" for label in labels],
        columns=[f"predicted_{CLASS_NAMES[label]}" for label in labels],
    ).to_csv(output_dir / f"{model_name}_confusion_matrix.csv")

    print(
        f"    overall: macro-F1={overall['f1_macro']:.4f}, "
        #f"balanced accuracy={overall['balanced_accuracy']:.4f}"
    )
    return summary, prediction_frame


def run_dataset(
    input_path: Path,
    models: dict[str, object],
    output_root: Path,
    folds: int,
    save_final_models: bool,
) -> list[dict[str, object]]:
    dataset_name = input_path.stem.removesuffix("_features_nogene")
    output_dir = output_root / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, row_ids = load_dataset(input_path)
    minimum_class_count = int(y.value_counts().min())
    #if folds < 2 or folds > minimum_class_count:
    #    raise ValueError(
    #        f"--folds must be between 2 and the smallest class count "
    #        f"({minimum_class_count}) for {input_path}."
    #    )

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(X, y))
    print(
        f"\n{dataset_name}: {len(X)} rows, {X.shape[1]} features, "
        f"class counts={y.value_counts().sort_index().to_dict()}"
    )

    summaries: list[dict[str, object]] = []
    prediction_frames: dict[str, pd.DataFrame] = {}
    for model_name, model in models.items():
        summary, prediction_frame = evaluate_model(
            model_name, model, X, y, row_ids, splits, output_dir
        )
        summaries.append({"dataset": dataset_name, **summary})
        prediction_frames[model_name] = prediction_frame

    summary_frame = pd.DataFrame(summaries).sort_values(
        ["f1_macro"], ascending=False
    )
    summary_frame.insert(1, "rank", range(1, len(summary_frame) + 1))
    summary_frame.to_csv(output_dir / "model_comparison.csv", index=False)

    best_model_name = str(summary_frame.iloc[0]["model"])
    prediction_frames[best_model_name].to_csv(
        output_dir / "best_model_predictions.csv", index=False
    )

    if save_final_models:
        models_dir = output_dir / "models"
        models_dir.mkdir(exist_ok=True)
        for model_name, model in models.items():
            print(f"  fitting final {model_name} model on all {dataset_name} rows")
            fitted = fit_model(model, X, y)
            joblib.dump(fitted, models_dir / f"{model_name}.joblib")

        metadata = {
            "input_file": str(input_path),
            "label_column": LABEL_COLUMN,
            "class_names": CLASS_NAMES,
            "feature_columns": X.columns.tolist(),
            "best_cross_validated_model": best_model_name,
            "selection_metric": "f1_macro",
            "random_state": RANDOM_STATE,
            "folds": folds,
        }
        (models_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    print(f"  best model: {best_model_name}")
    return summaries


def main() -> None:
    args = parse_args()
    models = build_models()
    if args.models:
        models = {name: models[name] for name in args.models}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries: list[dict[str, object]] = []
    for input_path in args.inputs:
        all_summaries.extend(
            run_dataset(
                input_path=input_path,
                models=models,
                output_root=args.output_dir,
                folds=args.folds,
                save_final_models=args.save_final_models,
            )
        )

    pd.DataFrame(all_summaries).sort_values(
        ["dataset", "f1_macro"],
        ascending=[True, False],
    ).to_csv(args.output_dir / "all_datasets_model_comparison.csv", index=False)
    print(f"\nFinished. Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
