"""Audio analysis: tempo, onset energy, hits, kick/snare bands, beat grid."""
import librosa
import numpy as np


def estimate_tempo_raw(y, sr):
    """librosa estimate + octave disambiguation (it often halves 200 BPM->99)."""
    est, _ = librosa.beat.beat_track(y=y, sr=sr)
    est = float(np.atleast_1d(est)[0])
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    onset = onset - onset.mean()
    ac = np.correlate(onset, onset, mode="full")[len(onset) - 1 :]
    ac = ac / (ac[0] or 1.0)
    fps_e = sr / 512
    best, best_v = est, -1.0
    for bpm in range(90, 261):
        lag = int(round(fps_e * 60.0 / bpm))
        if lag < len(ac):
            v = float(ac[lag])
            # Prefer the faster octave when it has real support.
            if v > best_v + (0.05 if bpm > 1.6 * est else 0.0):
                best, best_v = bpm, v
    return best


def analyze_audio(path, fps, max_duration=None, tempo=None):
    y, sr = librosa.load(path, sr=22050, mono=True, duration=max_duration)
    dur = len(y) / sr
    n_frames = max(1, int(dur * fps))
    if tempo is None:
        tempo = estimate_tempo_raw(y, sr)
        tempo_source = "estimated"
    else:
        tempo_source = "manual"
    # Onset envelope at frame resolution.
    hop = 512
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    times = librosa.frames_to_time(np.arange(len(onset)), sr=sr, hop_length=hop)
    frame_times = np.arange(n_frames) / fps
    energy = np.interp(frame_times, times, onset).astype(np.float32)
    # Normalize 0..1 (robust to outliers).
    hi = float(np.quantile(energy, 0.98)) or 1.0
    energy = np.clip(energy / hi, 0, 1)
    # Smooth with small moving average.
    k = max(1, int(fps * 0.08))
    energy = np.convolve(energy, np.ones(k) / k, mode="same").astype(np.float32)
    # Hit picking: peaks of onset envelope, fallback to energy threshold.
    peaks = librosa.util.peak_pick(
        energy, pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=0.25, wait=5
    )
    if len(peaks) == 0:
        peaks = np.where(energy > 0.65)[0]
        peaks = peaks[np.concatenate([[True], np.diff(peaks) > int(fps * 0.25)])]
    # Beat-grid kick detection: with a known tempo, score every beat slot so
    # fast kick runs (e.g. 200 BPM = kick every 0.3s) are all caught.
    beat = 60.0 / float(tempo)
    anchor = float(peaks[0] / fps) if len(peaks) else 0.0
    grid_hits = []
    k = 0
    while True:
        g = anchor + k * beat
        if g >= dur:
            break
        if g >= 0:
            lo = max(0, int((g - 0.07) * fps))
            hi = min(n_frames, int((g + 0.07) * fps) + 1)
            if hi > lo:
                m = lo + int(np.argmax(energy[lo:hi]))
                if float(energy[m]) >= 0.30:
                    grid_hits.append(m)
        k += 1
    if grid_hits:
        merged = sorted(set(peaks.tolist()) | set(grid_hits))
        # Dedupe within 0.1s, keep the stronger frame.
        dedup, last = [], -10**9
        for p in merged:
            if p - last > int(fps * 0.1):
                dedup.append(p)
                last = p
            elif float(energy[p]) > float(energy[last]):
                dedup[-1] = p
                last = p
        peaks = np.array(sorted(dedup), dtype=int)
    strengths = energy[peaks] if len(peaks) else np.array([], dtype=np.float32)
    # Kick vs snare split by frequency band: kicks live <150 Hz, snares bite
    # 150-500 Hz. Per-band spectral flux -> normalized envelopes.
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    t_spec = librosa.frames_to_time(np.arange(S.shape[1] - 1), sr=sr, hop_length=512)

    def band_flux(flo, fhi):
        B = S[(freqs >= flo) & (freqs < fhi), :]
        d = np.diff(B, axis=1)
        d[d < 0] = 0.0
        return d.sum(axis=0)

    def to_frame_env(flux, smooth_s=0.03):
        env = np.interp(frame_times, t_spec, flux).astype(np.float32)
        k = max(1, int(fps * smooth_s))
        env = np.convolve(env, np.ones(k) / k, mode="same").astype(np.float32)
        hi = float(np.quantile(env, 0.98)) or 1.0
        return np.clip(env / hi, 0, 1)

    kick_env = to_frame_env(band_flux(30, 150))
    snare_env = to_frame_env(band_flux(150, 500))
    # Hit decay envelope for pulse effects.
    hit_env = np.zeros(n_frames, dtype=np.float32)
    is_hit = np.zeros(n_frames, dtype=bool)
    is_hit[peaks] = True
    decay = np.exp(-np.arange(n_frames) / (fps * 0.25))
    for p, s in zip(peaks.tolist(), strengths.tolist()):
        w = min(n_frames - p, len(decay))
        hit_env[p : p + w] = np.maximum(hit_env[p : p + w], decay[:w] * float(s))
    tempo = float(tempo)
    # Beat grid with per-beat kick/snare strengths (max band-flux near slot).
    beat_dur = 60.0 / tempo
    anchor = float(peaks[0] / fps) if len(peaks) else 0.0
    beats = []
    k = 0
    while True:
        bt = anchor + k * beat_dur
        if bt >= dur:
            break
        if bt >= 0:
            lo = max(0, int((bt - 0.06) * fps))
            hi = min(n_frames, int((bt + 0.06) * fps) + 1)
            if hi > lo:
                ks = float(kick_env[lo:hi].max())
                ss = float(snare_env[lo:hi].max())
            else:
                ks, ss = 0.0, 0.0
            beats.append([round(bt, 4), round(ks, 4), round(ss, 4)])
        k += 1
    return {
        "y": y,
        "sr": sr,
        "dur": dur,
        "n_frames": n_frames,
        "energy": energy,
        "hit_env": hit_env,
        "kick_env": kick_env,
        "snare_env": snare_env,
        "is_hit": is_hit,
        "hits": sorted((float(p / fps), float(s)) for p, s in zip(peaks.tolist(), strengths.tolist())),
        "beats": beats,
        "beat_dur": beat_dur,
        "t0": anchor,
        "tempo": float(tempo),
        "tempo_source": tempo_source,
    }
