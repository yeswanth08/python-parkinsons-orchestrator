import numpy as np

# recurrence period density entropy
def compute_rpde(signal: np.ndarray, m: int = 4, tau: int = 1, T_max: int = 200, target_sr: int = 8000, original_sr: int = 22050) -> float:
    try:
        step = max(1, round(original_sr / target_sr))
        signal = signal[::step]
        n = len(signal)
        max_lag = (m - 1) * tau
        if n < max_lag + T_max + 10:
            return np.nan

        embedded = np.array(
            [signal[i: n - max_lag + i] for i in range(0, max_lag + 1, tau)]
        ).T
        N_embed = len(embedded)

        rng = np.random.default_rng(42)
        probe_idx = rng.choice(N_embed, size=min(300, N_embed), replace=False)
        probe = embedded[probe_idx]
        diff = probe[:, None, :] - probe[None, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=-1))
        flat = dists[dists > 0]
        if len(flat) == 0:
            return np.nan
        epsilon = float(np.percentile(flat, 10))

        # vectorized -> all seeds x all timesteps at once 
        n_seeds = min(400, N_embed - T_max - 1)
        seed_idx = rng.choice(N_embed - T_max - 1, size=n_seeds, replace=False)

        # shape -> (n_seeds, T_max, m)
        trajectories = np.stack(
            [embedded[seed_idx + t] for t in range(T_max + 1)], axis=1
        )
        origins = trajectories[:, 0:1, :]               # (n_seeds, 1, m)
        dists_all = np.sqrt(((trajectories - origins) ** 2).sum(axis=-1))  # (n_seeds, T_max+1)
        inside = dists_all <= epsilon                    # (n_seeds, T_max+1)

        # first return -> was inside at t=0, left, then came back
        periods = []
        for i in range(n_seeds):
            row = inside[i, 1:]                          # skip t=0
            left = np.argmax(~row) if not row.all() else -1
            if left == -1:
                continue
            returns = np.where(row[left:])[0]
            if len(returns) > 0:
                periods.append(int(left + returns[0] + 1))

        if len(periods) < 10:
            return np.nan

        n_bins = min(T_max, max(len(set(periods)), 2))
        hist, _ = np.histogram(periods, bins=n_bins)
        hist = hist / (hist.sum() + 1e-12)
        hist = hist[hist > 0]
        H = float(-np.sum(hist * np.log(hist)))
        H_max = float(np.log(T_max))
        return H / H_max

    except Exception:
        return np.nan

# detrended fluctutation analysis
def compute_dfa(signal: np.ndarray, n_scales: int = 16) -> float:
    try:
        n = len(signal)
        y = np.cumsum(signal - np.mean(signal))

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
            # reshape into segments — no per-segment loop
            chunks = y[:segs * s].reshape(segs, s)          
            x = np.arange(s, dtype=np.float64)
            # vectorized linear detrend across all segments at once
            '''
                here eliminated per seg polyfit loop which is overkilling the time complexity 
            '''
            xm = x - x.mean()
            ym = chunks - chunks.mean(axis=1, keepdims=True)
            slope = (ym * xm).sum(axis=1) / (xm * xm).sum()  
            trend = slope[:, None] * xm + chunks.mean(axis=1, keepdims=True)
            rms = np.sqrt(np.mean((chunks - trend) ** 2, axis=1))
            fluct.append(float(rms.mean()))

        if len(fluct) < 2:
            return np.nan

        log_s = np.log(scales[:len(fluct)].astype(float))
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
        idx = rng.choice(N, size=min(300, N), replace=False)
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
        