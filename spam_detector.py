from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, cast

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent


def load_dataset(dataset_dir: Path) -> pd.DataFrame:
    csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {dataset_dir}")

    frames = []
    for file_path in csv_files:
        frame = pd.read_csv(file_path)
        frame = frame[["CONTENT", "CLASS"]].dropna()
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data["CONTENT"] = data["CONTENT"].astype(str)
    data["CLASS"] = data["CLASS"].astype(int)
    return data


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                ),
            ),
            ("clf", ComplementNB(alpha=0.5)),
        ]
    )


def train_and_evaluate(
    data: pd.DataFrame, test_size: float, random_state: int
) -> tuple[Pipeline, str]:
    x_train, x_test, y_train, y_test = train_test_split(
        data["CONTENT"],
        data["CLASS"],
        test_size=test_size,
        random_state=random_state,
        stratify=data["CLASS"],
    )

    model = build_pipeline()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    report = cast(
        str,
        classification_report(
            y_test,
            y_pred,
            target_names=["Ham", "Spam"],
            digits=4,
            output_dict=False,
        ),
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Ham", "Spam"],
        cmap="Blues",
    )

    return model, report


def predict_texts(model: Pipeline, texts: Iterable[str]) -> list[int]:
    return model.predict(list(texts)).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YouTube spam detector trainer")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=BASE_DIR / "youtube-dataset",
        help="Path to folder containing Youtube*.csv files",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=BASE_DIR / "artifacts" / "spam_model.joblib",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction used for test split",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=365,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--predict",
        nargs="*",
        default=[],
        help="Optional texts to classify after training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = load_dataset(args.dataset_dir)

    model, report = train_and_evaluate(
        data=data,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)

    print("Training finished.")
    print(f"Model saved to: {args.model_out}")
    print("\nClassification report:\n")
    print(report)

    if args.predict:
        predictions = predict_texts(model, args.predict)
        print("\nPredictions:")
        for text, label in zip(args.predict, predictions):
            mapped = "Spam" if label == 1 else "Ham"
            print(f"- {mapped}: {text}")


if __name__ == "__main__":
    main()