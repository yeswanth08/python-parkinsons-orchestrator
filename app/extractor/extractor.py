import numpy as np
import parselmouth

from parselmouth.praat import call
from typing import Optional
from app.extractor.non_linear_features import _compute_d2,_compute_dfa,_compute_ppe,_compute_rpde

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
        """
        Six nonlinear signal complexity measures from [L07, L08].

        spread1, spread2  [L08]
        ─────────────────
        Nonlinear measures of fundamental frequency variation computed from
        the voiced F0 samples. In [L08] these are described as the mean and
        standard deviation of the absolute log-pitch deviations.

        spread1 = mean( log(F0_i) )   over voiced frames
        spread2 = std ( log(F0_i) )   over voiced frames

        Higher values → more F0 spread → more dysphonic [L08].

        RPDE  [L07, W17]
        ────
        Recurrence Period Density Entropy — measures the periodicity of the
        vocal fold vibration signal in reconstructed phase space.

        Algorithm (Little et al. 2007):
          1. Delay-embed signal x_n into M-dimensional phase space:
             X_n = [x_n, x_{n+τ}, x_{n+2τ}, …, x_{n+(M-1)τ}]
          2. For each seed X_s, define an ε-ball (radius ε, chosen
             adaptively as the 10th percentile of pairwise distances).
          3. Track the time T between successive returns of the trajectory
             to the ε-ball after having left it.  Record all T in a histogram.
          4. Normalise histogram → P(T) (recurrence period density).
          5. RPDE = H(P) / log(T_max)  ∈ [0,1]
             where H(P) = −Σ P(T) log P(T)  (Shannon entropy).

        Healthy (periodic) → sharp P(T) peak → low RPDE ≈ 0.3.
        PD (irregular)     → flat P(T)       → high RPDE ≈ 0.6+ [L07].

        DFA  [L07]
        ───
        Detrended Fluctuation Analysis — measures the fractal self-similarity
        (long-range correlations) of the voice signal.

        Algorithm (Peng et al. 1994, applied to voice in [L07]):
          1. Integrate: Y(k) = Σ_{i=1}^{k} (x_i − x̄)
          2. Divide Y into non-overlapping windows of size s.
          3. Detrend each window (subtract linear fit).
          4. Compute RMS fluctuation F(s).
          5. Fit: log F(s) = α·log(s) + const.  → return α.

        Healthy voice: α ≈ 0.5–0.7  (near-random).
        PD voice:      α ≈ 0.6–0.9  (increased long-range correlation) [L07].

        D2  [L07]
        ──
        Correlation dimension — a measure of the geometric complexity of
        the attractor in phase space. Higher D2 → more complex/chaotic signal.
        For the UCI dataset a log-variance proxy is used [L07], consistent
        with the range of D2 values in the published dataset (≈ −5 to −6 for
        a peak-normalised signal).

        D2 = log( Var(signal) )

        PPE  [L08]
        ───
        Pitch Period Entropy — measures the entropy of the F0 distribution,
        introduced in [L08] to be robust to confounding environmental noise.

        PPE = −Σ p_k · log(p_k)

        where p_k is the k-th bin of the normalised histogram of log(F0_i).

        Low PPE → stable, regular F0 (healthy).
        High PPE → irregular, diffuse F0 (PD) [L08].
        """
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
            "RPDE": _compute_rpde(signal, original_sr=int(self.snd.sampling_frequency)),
            "DFA":     _compute_dfa(signal),
            "D2":      _compute_d2(signal),
            "PPE":     _compute_ppe(log_period),
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
    """
    PRIMARY ENTRY POINT.

    Extract all acoustic features from a .wav file.
    Returns a single flat dict whose keys cover every column in both
    CLASSIFICATION_FEATURES and SEVERITY_FEATURES (except age/sex/test_time).

    Usage
    -----
        features = extract_voice_features("vowel.wav")
        clf_vec  = build_classification_vector(features)
        sev_vec  = build_severity_vector(features, age=65, sex=0, test_time=0)
    """
    return VoiceFeatureExtractor(audio_path).extract_all()
