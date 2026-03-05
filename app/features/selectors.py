from app.schema.schema import CLASSIFICATION_FEATURES, SEVERITY_FEATURES

def build_classification_vector(feature_dict: dict) -> list:
    """
    Ordered list of 22 floats for the classification model.
    Uses CLASSIFICATION_FEATURES key order.
    """
    return [feature_dict[k] for k in CLASSIFICATION_FEATURES]


def build_severity_vector(
    feature_dict: dict,
    age: float,
    sex: int,
    test_time: float,
) -> list:
    """
    Ordered list for the severity/telemonitoring model.

    Uses SEVERITY_FEATURES (telemonitoring) key names — these differ from
    the classification schema:
        Jitter(%)     ≡  MDVP:Jitter(%)
        Shimmer:APQ11 ≡  MDVP:APQ
    All keys are present in the dict from extract_voice_features().

    Parameters
    ----------
    feature_dict : output of extract_voice_features()
    age          : patient age in years
    sex          : 0 = female, 1 = male
    test_time    : days since trial recruitment
    """
    return [
        age,
        sex,
        test_time,
        feature_dict["Jitter(%)"],       # telemonitoring key ≠ MDVP:Jitter(%)
        feature_dict["Jitter(Abs)"],
        feature_dict["Jitter:RAP"],
        feature_dict["Jitter:PPQ5"],
        feature_dict["Jitter:DDP"],
        feature_dict["Shimmer"],
        feature_dict["Shimmer(dB)"],
        feature_dict["Shimmer:APQ3"],
        feature_dict["Shimmer:APQ5"],
        feature_dict["Shimmer:APQ11"],   # telemonitoring key ≠ MDVP:APQ
        feature_dict["Shimmer:DDA"],
        feature_dict["NHR"],
        feature_dict["HNR"],
        feature_dict["RPDE"],
        feature_dict["DFA"],
        feature_dict["PPE"],
    ]