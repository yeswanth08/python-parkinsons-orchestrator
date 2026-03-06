from app.schema.schema import CLASSIFICATION_FEATURES, SEVERITY_FEATURES

def build_classification_vector(feature_dict: dict) -> list:
    return [feature_dict[k] for k in CLASSIFICATION_FEATURES]


def build_severity_vector(feature_dict: dict,age: float,sex: int,test_time: float) -> list:
    return [
        age,
        sex,
        test_time,
        feature_dict["Jitter(%)"],       
        feature_dict["Jitter(Abs)"],
        feature_dict["Jitter:RAP"],
        feature_dict["Jitter:PPQ5"],
        feature_dict["Jitter:DDP"],
        feature_dict["Shimmer"],
        feature_dict["Shimmer(dB)"],
        feature_dict["Shimmer:APQ3"],
        feature_dict["Shimmer:APQ5"],
        feature_dict["Shimmer:APQ11"],
        feature_dict["Shimmer:DDA"],
        feature_dict["NHR"],
        feature_dict["HNR"],
        feature_dict["RPDE"],
        feature_dict["DFA"],
        feature_dict["PPE"],
    ]