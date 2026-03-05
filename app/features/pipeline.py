import joblib
import pandas as pd

from pathlib import Path
from app.features.selectors import build_classification_vector,build_severity_vector
from app.schema.schema import CLASSIFICATION_FEATURES,SEVERITY_FEATURES

def run_pipeline(feature_dict, age, sex, test_time):
    # loading the ml models
    BASE_DIR = Path(__file__).resolve().parents[2]

    classifier = joblib.load(BASE_DIR / "models/classification_model.pkl")
    severity_model = joblib.load(BASE_DIR /"models/severity_model.pkl")

    # classification
    clf_vector = build_classification_vector(feature_dict)
    clf_df = pd.DataFrame([clf_vector],columns=CLASSIFICATION_FEATURES)

    # print(clf_df.to_dict(orient='records'))
    # print(classifier.predict_proba(clf_df))  # confidence

    prediction = classifier.predict(clf_df)[0]

    if prediction == 0:
        # status 0 for healthy
        return {"parkinsons": False}

    # severity
    sev_vector = build_severity_vector(feature_dict, age, sex, test_time)
    sev_df = pd.DataFrame([sev_vector],columns=SEVERITY_FEATURES)

    severity = severity_model.predict(sev_df)[0]

    return {
        "parkinsons": True,
        "severity": float(severity)
    }
