import joblib
import pandas as pd

from pathlib import Path
from app.helper.selectors import build_classification_vector,build_severity_vector
from app.schema.schema import CLASSIFICATION_FEATURES,SEVERITY_FEATURES

# loading the ml models on import this can save the loading of models from the disk
BASE_DIR = Path(__file__).resolve().parents[2]

classifier = joblib.load(BASE_DIR / "models/classification_model.pkl")
severity_model = joblib.load(BASE_DIR /"models/severity_model.pkl")

def run_pipeline(feature_dict, age, sex, test_time)->dict:
    # classification
    clf_vector = build_classification_vector(feature_dict)
    clf_df = pd.DataFrame([clf_vector],columns=CLASSIFICATION_FEATURES)

    # print(clf_df.to_dict(orient='records'))
    # print(classifier.predict_proba(clf_df))  # confidence

    prediction = classifier.predict(clf_df)[0]

    if prediction == 0:
        # status 0 for healthy
        return {
            "parkinsons": False,
            "severity": float(0)
        }

    # severity
    sev_vector = build_severity_vector(feature_dict, age, sex, test_time)
    sev_df = pd.DataFrame([sev_vector],columns=SEVERITY_FEATURES)

    severity = severity_model.predict(sev_df)[0]

    # extraction of the feature results for the report showcase
    response = {
        "parkinsons": True,
        "severity": round(float(severity),1),
        "extracted_voice_features": {},
    }

    clf_df = clf_df.astype(float).round(2)
    sev_df = sev_df.astype(float).round(2)


    for col in clf_df.columns:
        response["extracted_voice_features"][col] = float(clf_df.iloc[0][col])

    for col in sev_df.columns:
        response["extracted_voice_features"][col] = float(sev_df.iloc[0][col])

    # print(f"[Pipeline] predict_proba: {classifier.predict_proba(clf_df)}")
    # print(f"[Pipeline] prediction: {prediction}")
    # print(f"[Pipeline] full clf_vector: {clf_df.to_dict(orient='records')}")
    # print(f"[Pipeline] NaN features: {clf_df.columns[clf_df.isna().any()].tolist()}")
    # print(f"[Pipeline] clf_vector: {clf_df.iloc[0].to_dict()}")

    return response
