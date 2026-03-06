import numpy as np

# recurrence period density entropy
def compute_rpde(signal: np.ndarray,m: int = 4,tau: int = 1,T_max: int = 200,target_sr: int = 8000,original_sr: int = 22050) -> float:
    try:
        step = max(1, round(original_sr / target_sr))
        signal = signal[::step]

        n = len(signal)
        max_lag = (m - 1) * tau
        if n < max_lag + T_max + 10:
            return np.nan   

        # 1. Delay embedding — shape (N_embed, m)
        embedded = np.array(
            [signal[i: n - max_lag + i] for i in range(0, max_lag + 1, tau)]
        ).T
        N_embed = len(embedded)

        # 2. Adaptive ε from a random probe subsample
        rng       = np.random.default_rng(42)
        probe_idx = rng.choice(N_embed, size=min(300, N_embed), replace=False)
        probe     = embedded[probe_idx]
        diff      = probe[:, None, :] - probe[None, :, :]   # (P, P, m)
        dists     = np.sqrt((diff ** 2).sum(axis=-1))        # (P, P)
        flat      = dists[dists > 0]
        if len(flat) == 0:
            return np.nan
        epsilon = float(np.percentile(flat, 10))

        # 3. First-return periods
        n_seeds  = min(400, N_embed - T_max - 1)
        seed_idx = rng.choice(N_embed - T_max - 1, size=n_seeds, replace=False)
        periods  = []
        for s in seed_idx:
            x0     = embedded[s]
            inside = True                    # trajectory starts inside ball
            for t in range(1, T_max + 1):
                d = float(np.sqrt(np.sum((embedded[s + t] - x0) ** 2)))
                if inside:
                    if d > epsilon:
                        inside = False       # left the ball
                else:
                    if d <= epsilon:
                        periods.append(t)    # first return ✓
                        break

        if len(periods) < 10:
            return np.nan

        # 4. Normalised entropy
        n_bins  = min(T_max, max(len(set(periods)), 2))
        hist, _ = np.histogram(periods, bins=n_bins)
        hist    = hist / (hist.sum() + 1e-12)
        hist    = hist[hist > 0]
        H       = float(-np.sum(hist * np.log(hist)))
        # H_max   = float(np.log(len(hist))) if len(hist) > 1 else 1.0
        H_max = float(np.log(T_max))
        return H / H_max

    except Exception:
        return np.nan


# detrended fluctutation analysis
def compute_dfa(signal: np.ndarray, n_scales: int = 16) -> float:
    try:
        n = len(signal)
        y = np.cumsum(signal - np.mean(signal))   # integrated signal

        min_s, max_s = 4, n // 4
        if min_s >= max_s:
            return np.nan

        scales = np.unique(
            np.floor(
                np.logspace(np.log10(min_s), np.log10(max_s), n_scales)
            ).astype(int)
        )

        fluct = []
        for s in scales:
            segs = n // s
            if segs == 0:
                continue
            rms_vals = []
            for k in range(segs):
                seg   = y[k * s: (k + 1) * s]
                x_seg = np.arange(len(seg), dtype=np.float64)
                trend = np.polyval(np.polyfit(x_seg, seg, 1), x_seg)
                rms_vals.append(float(np.sqrt(np.mean((seg - trend) ** 2))))
            fluct.append(float(np.mean(rms_vals)))

        if len(fluct) < 2:
            return np.nan

        log_s = np.log(scales[: len(fluct)].astype(float))
        log_f = np.log(np.array(fluct) + 1e-12)
        return float(np.polyfit(log_s, log_f, 1)[0])

    except Exception:
        return np.nan

# co-rrelation dimension d2 via grassberger-procaccia algorithm
def compute_d2(signal: np.ndarray, m: int = 4, tau: int = 1) -> float:
    try:
        # Downsample to ~4000 points for speed
        step = max(1, len(signal) // 4000)
        x = signal[::step]
        n = len(x)
        if n < 100:
            return np.nan

        # Delay embedding
        max_lag = (m - 1) * tau
        embedded = np.array(
            [x[i: n - max_lag + i] for i in range(0, max_lag + 1, tau)]
        ).T  # shape (N_embed, m)
        N = len(embedded)

        # Subsample for pairwise distances
        rng = np.random.default_rng(42)
        idx = rng.choice(N, size=min(500, N), replace=False)
        sub = embedded[idx]

        # Pairwise distances
        diff = sub[:, None, :] - sub[None, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=-1)).flatten()
        dists = dists[dists > 0]
        if len(dists) == 0:
            return np.nan

        # Correlation integral C(r) at log-spaced r values
        r_vals = np.logspace(
            np.log10(np.percentile(dists, 5)),
            np.log10(np.percentile(dists, 50)),
            20
        )
        C = np.array([np.mean(dists < r) for r in r_vals])
        
        # Filter valid points
        valid = C > 0
        if valid.sum() < 4:
            return np.nan

        # D2 = slope of log C(r) vs log r
        slope, _ = np.polyfit(np.log(r_vals[valid]), np.log(C[valid]), 1)
        return float(np.clip(slope, 0.0, 10.0))

    except Exception:
        return np.nan

# pitch period entropy 
def compute_ppe(log_period: np.ndarray, bins: int = 50) -> float:
    try:
        if len(log_period) < 2:
            return np.nan
        median = np.median(log_period)
        semitone = np.log(2) / 12
        hist, _ = np.histogram(log_period, bins=bins,
                               range=(median - 3*semitone, median + 3*semitone))
        hist = hist / (hist.sum() + 1e-12)
        hist = hist[hist > 0]
        H     = float(-np.sum(hist * np.log(hist)))
        H_max = float(np.log(bins))  # normalise by log(total bins), not occupied
        return H / H_max
    except Exception:
        return np.nan
        