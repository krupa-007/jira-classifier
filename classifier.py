"""
classifier.py
ML classification logic for JIRA ticket categorisation.

Uses a TF-IDF vectoriser combined with a Logistic Regression model to predict
the category of a JIRA ticket from its summary and description text.
"""

import os
import pickle
import re
import string

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
LABEL_ENCODER_PATH = os.path.join(os.path.dirname(__file__), "label_encoder.pkl")

CATEGORIES = ["Bug", "Feature", "Task"]


# ---------------------------------------------------------------------------
# Text pre-processing
# ---------------------------------------------------------------------------


def _clean_text(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace from *text*."""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)          # remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _combine_fields(row: pd.Series) -> str:
    """Combine the summary and description columns into a single string."""
    summary = str(row.get("summary", ""))
    description = str(row.get("description", ""))
    return _clean_text(f"{summary} {description}")


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------


def build_pipeline() -> Pipeline:
    """Return a fresh, untrained sklearn Pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=10_000,
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train(df: pd.DataFrame) -> dict:
    """Train the classifier on *df* and persist the model to disk.

    Parameters
    ----------
    df:
        DataFrame with columns ``summary``, ``description``, and ``category``.

    Returns
    -------
    dict
        A dictionary containing ``accuracy``, ``report``, and ``confusion_matrix``.
    """
    if df.empty:
        raise ValueError("Training dataframe must not be empty.")

    required = {"summary", "description", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["category"])

    X = df.apply(_combine_fields, axis=1)
    y = df["category"].str.strip()

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = float(np.mean(y_pred == y_test))
    report = classification_report(
        y_test, y_pred, target_names=le.classes_, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    # Persist
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    return {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm,
        "classes": list(le.classes_),
    }


def load_model() -> tuple:
    """Load the persisted pipeline and label encoder from disk.

    Returns
    -------
    tuple
        ``(pipeline, label_encoder)``

    Raises
    ------
    FileNotFoundError
        If no trained model exists on disk.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_ENCODER_PATH):
        raise FileNotFoundError(
            "No trained model found. Train the model first using train()."
        )
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)
    with open(LABEL_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    return pipeline, le


def predict_single(summary: str, description: str) -> dict:
    """Predict the category for a single JIRA ticket.

    Parameters
    ----------
    summary:
        Ticket summary / title.
    description:
        Ticket description body.

    Returns
    -------
    dict
        ``{"category": str, "probabilities": dict}``
    """
    pipeline, le = load_model()

    text = _clean_text(f"{summary} {description}")
    proba = pipeline.predict_proba([text])[0]
    idx = int(np.argmax(proba))
    category = le.inverse_transform([idx])[0]
    probabilities = {cls: float(p) for cls, p in zip(le.classes_, proba)}

    return {"category": category, "probabilities": probabilities}


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Predict categories for a batch of JIRA tickets.

    Parameters
    ----------
    df:
        DataFrame with at least ``summary`` and ``description`` columns.

    Returns
    -------
    pd.DataFrame
        Input dataframe with an additional ``predicted_category`` column.
    """
    pipeline, le = load_model()

    texts = df.apply(_combine_fields, axis=1)
    y_pred = pipeline.predict(texts)
    categories = le.inverse_transform(y_pred)

    result = df.copy()
    result["predicted_category"] = categories
    return result


def is_model_trained() -> bool:
    """Return ``True`` if a trained model exists on disk."""
    return os.path.exists(MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH)
