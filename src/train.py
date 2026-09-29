
from pathlib import Path
import json
import joblib

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"


def train_model():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,
        random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model Accuracy: {accuracy:.4f}")

    joblib.dump(model, ARTIFACT_DIR / "iris_model.joblib")

    metrics = {
        "accuracy": float(accuracy),
        "test_samples": int(len(y_test))
    }

    (ARTIFACT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=4)
    )

    print("Model and metrics saved successfully!")
    return accuracy


if __name__ == "__main__":
    train_model()