import numpy as np
import parselmouth

from parselmouth.praat import call
from typing import Optional
from app.extractor.non_linear_features import compute_d2,compute_dfa,compute_ppe,compute_rpde

class VoiceFeatureExtractor:
    def __init__(self,audio_path: str,pitch_floor: float = 75.0,pitch_ceiling: float = 500.0):
        self.audio_path    = audio_path
        self.pitch_floor   = pitch_floor
        self.pitch_ceiling = pitch_ceiling

        self.snd = parselmouth.Sound(audio_path)
        self._preprocess()

        # ── Praat objects — computed once, reused across all methods ──
        #
        # Cross-correlation pitch: matches MDVP's autocorrelation method
        # that was used to produce the UCI features [L08, P96].
        self.pitch = call(self.snd, "To Pitch", 0.0, self.pitch_floor, self.pitch_ceiling)

        # Periodic PointProcess: required for jitter and shimmer [P96].
        # Uses cross-correlation (cc) consistent with MDVP [P96 Voice§2].
        self.point_process = call(self.snd,"To PointProcess (periodic, cc)",self.pitch_floor,self.pitch_ceiling)

        # Harmonicity via cc: matches HNR calculation method in [L07, P96].
        self.harmonicity = call(self.snd,"To Harmonicity (cc)",0.01,self.pitch_floor,0.1,1.0)

    def _preprocess(self):
        # Converting to mono and peak-normalise to [-1, 1].
        self.snd = self.snd.convert_to_mono()
        peak = np.max(np.abs(self.snd.values))
        if peak > 0:
            self.snd.values /= peak

    def _pitch_features(self) -> dict:
        # Extracting mean, max, and min of the fundamental frequency F0.
        p = self.pitch
        return {
            "MDVP:Fo(Hz)":  _safe(call, p, "Get mean",    0, 0, "Hertz"),
            "MDVP:Fhi(Hz)": _safe(call, p, "Get maximum", 0, 0, "Hertz", "Parabolic"),
            "MDVP:Flo(Hz)": _safe(call, p, "Get minimum", 0, 0, "Hertz", "Parabolic"),
        }

    def _jitter_features(self) -> dict:
        pp   = self.point_process
        args = (0, 0, 0.0001, 0.02, 1.3)
        local = _safe(call, pp, "Get jitter (local)",           *args)
        abs_  = _safe(call, pp, "Get jitter (local, absolute)", *args)
        rap   = _safe(call, pp, "Get jitter (rap)",             *args)
        ppq5  = _safe(call, pp, "Get jitter (ppq5)",            *args)
        ddp   = _safe(call, pp, "Get jitter (ddp)",             *args)

        return {
            # Classification (MDVP) schema keys
            "MDVP:Jitter(%)":   local,
            "MDVP:Jitter(Abs)": abs_,
            "MDVP:RAP":         rap,
            "MDVP:PPQ":         ppq5,
            "Jitter:DDP":       ddp,     # identical key in both schemas
            # Telemonitoring schema keys [T10] — same values, different names
            "Jitter(%)":        local,
            "Jitter(Abs)":      abs_,
            "Jitter:RAP":       rap,
            "Jitter:PPQ5":      ppq5,
        }

    def _shimmer_features(self) -> dict:
        snd  = self.snd
        pp   = self.point_process
        args = (0, 0, 0.0001, 0.02, 1.3, 1.6)

        local  = _safe(call, [snd, pp], "Get shimmer (local)",    *args)
        db_    = _safe(call, [snd, pp], "Get shimmer (local_dB)", *args)
        apq3   = _safe(call, [snd, pp], "Get shimmer (apq3)",     *args)
        apq5   = _safe(call, [snd, pp], "Get shimmer (apq5)",     *args)
        apq11  = _safe(call, [snd, pp], "Get shimmer (apq11)",    *args)
        dda    = _safe(call, [snd, pp], "Get shimmer (dda)",      *args)

        return {
            # Classification (MDVP) schema keys
            "MDVP:Shimmer":     local,
            "MDVP:Shimmer(dB)": db_,
            "Shimmer:APQ3":     apq3,   # shared key in both schemas
            "Shimmer:APQ5":     apq5,   # shared key in both schemas
            "MDVP:APQ":         apq11,
            "Shimmer:DDA":      dda,    # shared key in both schemas
            # Telemonitoring schema keys [T10]
            "Shimmer":          local,
            "Shimmer(dB)":      db_,
            "Shimmer:APQ11":    apq11,
        }

#   noise ration extractions
    def _harmonic_features(self) -> dict:
        hnr = _safe(call, self.harmonicity, "Get mean", 0, 0)
        if np.isnan(hnr):
            hnr = 0.0
        nhr = float(1.0 / (10.0 ** (hnr / 10.0) + 1e-9))
        return {"HNR": float(hnr), "NHR": nhr}

    def _nonlinear_features(self) -> dict:
        # signal     = self.snd.values[0].astype(np.float64)
        signal = self.snd.values.flatten().astype(np.float64)
        pitch_vals = self.pitch.selected_array["frequency"]
        pitch_vals = pitch_vals[pitch_vals > 0]

        if len(pitch_vals) < 10:
            return {k: np.nan for k in
                    ("RPDE", "DFA", "spread1", "spread2", "D2", "PPE")}

        # log_pitch = np.log(pitch_vals + 1e-9)

        log_period = np.log(1.0 / (pitch_vals + 1e-9))
        return {
            "spread1": float(np.mean(log_period)),   # explicit float never do np.float64
            "spread2": float(np.std(log_period)),
            # "RPDE":    _compute_rpde(signal),
            "RPDE":     compute_rpde(signal, original_sr=int(self.snd.sampling_frequency)),
            "DFA":     compute_dfa(signal),
            "D2":      compute_d2(signal),
            "PPE":     compute_ppe(log_period),
        }

    def extract_all(self) -> dict:
        # returns all the features in a bulk for classification and severity combined
        out: dict = {}
        out.update(self._pitch_features())
        out.update(self._jitter_features())
        out.update(self._shimmer_features())
        out.update(self._harmonic_features())
        out.update(self._nonlinear_features())
        return out

def _safe(fn, *args, fallback: float = np.nan, **kwargs) -> float:
    try:
        result = fn(*args, **kwargs)
        if result is None:
            return fallback
        result = float(result)
        return fallback if np.isnan(result) else result
    except Exception:
        return fallback


def extract_voice_features(audio_path: str) -> dict:
    return VoiceFeatureExtractor(audio_path).extract_all()
