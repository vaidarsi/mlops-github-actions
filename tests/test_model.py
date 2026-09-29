import json
from pathlib import Path

import joblib
from src.train import train_model

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "iris_model.joblib"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"


def test_model_training():
    accuracy = train_model()
    assert accuracy >= 0.80


def test_model_file_exists():
    assert MODEL_PATH.exists()


def test_model_prediction():
    model = joblib.load(MODEL_PATH)
    prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])

    assert len(prediction) == 1
    assert prediction[0] in [0, 1, 2]


def test_metrics_file():
    metrics = json.loads(METRICS_PATH.read_text())
    assert metrics["accuracy"] >= 0.80