"""Train TF-IDF + Logistic Regression, Naive Bayes, Random Forest (college stack)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.nlp import clean_text  # noqa: E402

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DATASET_URLS = [
    "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv",
]


def load_isot_csvs() -> pd.DataFrame | None:
    true_path = DATA_DIR / "True.csv"
    fake_path = DATA_DIR / "Fake.csv"
    if true_path.exists() and fake_path.exists():
        true_df = pd.read_csv(true_path)
        fake_df = pd.read_csv(fake_path)
        true_df["label"] = "REAL"
        fake_df["label"] = "FAKE"
        return pd.concat([true_df, fake_df], ignore_index=True)
    return None


def download_public_csv() -> pd.DataFrame:
    last_error = None
    for url in DATASET_URLS:
        try:
            df = pd.read_csv(url)
            out = DATA_DIR / "fake_or_real_news.csv"
            DATA_DIR.mkdir(exist_ok=True)
            df.to_csv(out, index=False)
            print(f"Downloaded dataset from {url} ({len(df)} rows)")
            return df
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"Could not download {url}: {exc}")
    raise RuntimeError(f"Could not load a training dataset: {last_error}")


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "text" not in df.columns and "title" in df.columns:
        df = df.copy()
        df["text"] = df["title"]
    if "text" not in df.columns:
        raise ValueError(f"No text column in {list(df.columns)}")
    if "title" in df.columns:
        df = df.copy()
        df["text"] = df["title"].fillna("").astype(str) + " " + df["text"].fillna("").astype(str)
    labels = df["label"].astype(str).str.upper().str.strip()
    labels = labels.replace({"TRUE": "REAL", "FALSE": "FAKE", "0": "REAL", "1": "FAKE"})
    df = df.assign(label=labels, text=df["text"].astype(str).map(clean_text))
    df = df[df["text"].str.len() > 40]
    df = df[df["label"].isin(["REAL", "FAKE"])]
    return df[["text", "label"]].dropna()


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    local = DATA_DIR / "fake_or_real_news.csv"
    df = load_isot_csvs()
    if df is None and local.exists():
        df = pd.read_csv(local)
        print(f"Loaded {local}")
    if df is None:
        df = download_public_csv()

    df = prepare_frame(df)
    print(df["label"].value_counts().to_string())

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
    )
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    models = {
        "logistic_regression": LogisticRegression(max_iter=400, class_weight="balanced"),
        "naive_bayes": MultinomialNB(),
        "random_forest": RandomForestClassifier(
            n_estimators=160,
            max_depth=40,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        ),
    }

    reports = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(x_train_vec, y_train)
        pred = model.predict(x_test_vec)
        reports[name] = classification_report(y_test, pred, output_dict=True)
        print(classification_report(y_test, pred))
        print("Confusion matrix:\n", confusion_matrix(y_test, pred))

    classes = list(models["logistic_regression"].classes_)
    fake_index = classes.index("FAKE")

    bundle = {
        "vectorizer": vectorizer,
        "models": models,
        "classes": classes,
        "fake_index": fake_index,
    }
    joblib.dump(bundle, MODELS_DIR / "ensemble.joblib")
    metrics_path = MODELS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"Saved {MODELS_DIR / 'ensemble.joblib'}")
    print(f"Saved {metrics_path}")


if __name__ == "__main__":
    main()
