#!/usr/bin/env python3
"""
Incremental Option D+ audio intensity analyzer for Mux assets.
Combines YAMNet vocal event tagging with DSP features:
  - Sustained voice-band loudness (200-3500 Hz)
  - Pitch height + voicing strength (vectorized autocorrelation)
  - Voicing-gated spectral centroid / brightness
  - Low-mid rhythmic periodicity (envelope onset autocorrelation, 0.8-5 Hz) + flux density
  - Multi-second continuity boost + spike rejection
Outputs 2 Hz compact JSON to public/intensity/<playbackId>.json.
"""
import argparse
import json
import os
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt

ROOT = Path(__file__).resolve().parent.parent
ASSETS_PATH = ROOT / "public" / "assets.json"
OUT_DIR = ROOT / "public" / "intensity"
MODEL_DIR = ROOT / ".cache" / "yamnet"

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/latest/yamnet.tflite"
LABELS_URL = "https://storage.googleapis.com/mediapipe-tasks/audio_classifier/yamnet_label_list.txt"

SR = 16000
OUT_HZ = 2  # 2 points/sec (0.5s resolution) for crisp scrubbing
SCHEMA_VERSION = 2

VOCAL_WEIGHTS = {
    "Shout": 1.0, "Yell": 1.0, "Bellow": 1.0, "Whoop": 1.0, "Screaming": 1.0,
    "Cheering": 1.0, "Groan": 1.0, "Grunt": 1.0, "Gasp": 1.0, "Pant": 1.0,
    "Wail, moan": 1.0, "Whimper": 0.9, "Laughter": 0.85, "Giggle": 0.8,
    "Crying, sobbing": 0.8,
    "Breathing": 0.65, "Sigh": 0.6, "Snort": 0.55, "Chuckle, chortle": 0.6,
    "Speech": 0.25, "Conversation": 0.2, "Chatter": 0.2, "Whispering": 0.15,
}

PENALTY_CLASSES = {
    "Music", "Musical instrument", "Singing", "Television", "Radio",
    "Static", "White noise", "Pink noise", "Rustle", "Rustling leaves",
}


def ensure_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "yamnet.tflite"
    labels_path = MODEL_DIR / "yamnet_label_list.txt"
    if not model_path.exists():
        print("Downloading YAMNet TFLite model...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
    if not labels_path.exists():
        urllib.request.urlretrieve(LABELS_URL, labels_path)
    labels = labels_path.read_text(encoding="utf-8").splitlines()
    return model_path, labels


def robust_norm(x, low=5.0, high=97.0):
    if len(x) == 0:
        return x.astype(np.float32)
    lo, hi = np.percentile(x, [low, high])
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def fetch_audio_pcm(playback_id: str) -> np.ndarray:
    """Stream 16kHz mono float32 PCM directly from Mux audio endpoint."""
    urls = [
        (f"https://stream.mux.com/{playback_id}/audio.m4a", []),
        (f"https://stream.mux.com/{playback_id}.m3u8", ["-map", "0:a:0"]),
    ]
    last_err = ""
    for url, extra in urls:
        cmd = [
            "ffmpeg", "-nostdin", "-loglevel", "error",
            "-i", url, *extra,
            "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode == 0 and len(proc.stdout) >= SR * 4:
            return np.frombuffer(proc.stdout, dtype=np.float32).copy()
        last_err = proc.stderr.decode("utf-8", errors="ignore").strip()
    raise RuntimeError(f"ffmpeg failed for {playback_id}: {last_err}")


def compute_yamnet_vocal(audio: np.ndarray, interpreter: Interpreter, labels: list[str], t_grid: np.ndarray) -> np.ndarray:
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    win_len = int(inp["shape"][-1])  # 15600 samples (~0.975s)
    hop_len = SR // 2                # 0.5s hop (2 Hz)

    w_vocal = np.zeros(len(labels), dtype=np.float32)
    w_penalty = np.zeros(len(labels), dtype=np.float32)
    for idx, name in enumerate(labels):
        if name in VOCAL_WEIGHTS:
            w_vocal[idx] = VOCAL_WEIGHTS[name]
        if name in PENALTY_CLASSES:
            w_penalty[idx] = 1.0

    times = []
    scores = []

    for start in range(0, len(audio), hop_len):
        chunk = audio[start : start + win_len]
        if len(chunk) < win_len:
            chunk = np.pad(chunk, (0, win_len - len(chunk)))
        tensor = chunk.reshape(inp["shape"]).astype(np.float32)
        interpreter.set_tensor(inp["index"], tensor)
        interpreter.invoke()
        probs = interpreter.get_tensor(out["index"]).flatten()

        vocal_peak = float(np.max(probs * w_vocal))
        vocal_sum = float(np.clip(np.sum(probs * w_vocal), 0.0, 1.5))
        noise_pen = float(np.clip(np.sum(probs * w_penalty), 0.0, 1.0))

        v = (0.65 * vocal_peak + 0.35 * min(vocal_sum, 1.0)) * (1.0 - 0.65 * noise_pen)
        times.append((start + win_len * 0.5) / SR)
        scores.append(v)

    y_2hz = np.interp(t_grid, times, scores).astype(np.float32)
    return median_filter(y_2hz, size=3)  # 1.5s median kills single-frame blips


def compute_vocal_dsp(audio: np.ndarray, t_grid: np.ndarray):
    """
    Computes at 20 Hz (then pooled to OUT_HZ):
      - Sustained voice-band loudness (200-3500 Hz)
      - Voicing strength (harmonicity via normalized autocorrelation)
      - Pitch height (100-695 Hz, credited when voiced)
      - Spectral centroid / brightness (200-3800 Hz)
    """
    sos_v = butter(4, [150, 3800], btype="bandpass", fs=SR, output="sos")
    v_audio = sosfiltfilt(sos_v, audio).astype(np.float32)

    frame_len = 1024   # 64 ms
    hop = 800          # 50 ms -> 20 Hz
    n_frames = max(1, (len(v_audio) - frame_len) // hop + 1)
    hann = np.hanning(frame_len).astype(np.float32)

    freqs = np.fft.rfftfreq(frame_len, d=1.0 / SR)
    band_mask = (freqs >= 200.0) & (freqs <= 3800.0)
    band_freqs = freqs[band_mask].astype(np.float32)

    # Autocorrelation lag range for F0 = 100 Hz .. 695 Hz
    lag_min, lag_max = 23, 160

    rms_arr = np.zeros(n_frames, dtype=np.float32)
    voicing_arr = np.zeros(n_frames, dtype=np.float32)
    pitch_arr = np.zeros(n_frames, dtype=np.float32)
    centroid_arr = np.zeros(n_frames, dtype=np.float32)

    # Process in chunks to keep RAM tiny even on multi-hour videos
    chunk_size = 2048
    for f0 in range(0, n_frames, chunk_size):
        f1 = min(n_frames, f0 + chunk_size)
        b = f1 - f0
        starts = np.arange(f0, f1) * hop
        idx = starts[:, None] + np.arange(frame_len)[None, :]
        idx = np.clip(idx, 0, len(v_audio) - 1)
        frames = v_audio[idx]

        # RMS energy
        rms_arr[f0:f1] = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-10)

        wframes = frames * hann[None, :]

        # Spectral centroid in 200-3800 Hz
        spec_p = np.abs(np.fft.rfft(wframes, n=frame_len, axis=1)) ** 2
        bp = spec_p[:, band_mask]
        centroid_arr[f0:f1] = np.sum(bp * band_freqs[None, :], axis=1) / (np.sum(bp, axis=1) + 1e-10)

        # Normalized autocorrelation via FFT (padded to 2048 to avoid wrap-around)
        fft_acf = np.fft.rfft(wframes, n=2048, axis=1)
        acf = np.fft.irfft(np.abs(fft_acf) ** 2, n=2048, axis=1)[:, : lag_max + 2]
        acf_norm = acf / (acf[:, 0:1] + 1e-10)

        # Keep only true local maxima inside [lag_min, lag_max] (never the main-lobe slope)
        mid = acf_norm[:, lag_min : lag_max + 1]
        left = acf_norm[:, lag_min - 1 : lag_max]
        right = acf_norm[:, lag_min + 1 : lag_max + 2]
        is_peak = (mid >= left) & (mid >= right) & (mid > 0.0)
        masked = np.where(is_peak, mid, 0.0)

        best_rel = np.argmax(masked, axis=1)
        best_val = masked[np.arange(b), best_rel]
        best_lag = lag_min + best_rel

        voicing_arr[f0:f1] = np.clip(best_val, 0.0, 1.0)
        # Log-scaled pitch height (0..1 between ~110 Hz and ~650 Hz), gated by voicing
        hz = SR / best_lag.astype(np.float32)
        pitch_norm = np.clip((np.log2(hz) - np.log2(110.0)) / (np.log2(650.0) - np.log2(110.0)), 0.0, 1.0)
        pitch_arr[f0:f1] = np.where(best_val > 0.30, pitch_norm * best_val, 0.0)

    t_20 = (np.arange(n_frames, dtype=np.float32) * hop + frame_len * 0.5) / SR

    # 1.5s rolling median (30 frames @ 20 Hz) suppresses brief transient pops
    loud_20 = median_filter(np.log1p(rms_arr * 25.0), size=30)
    voic_20 = median_filter(voicing_arr, size=20)
    pitch_20 = median_filter(pitch_arr, size=20)
    # Gate spectral brightness by voicing so fabric rustle doesn't score high
    bright_raw = np.clip((centroid_arr - 500.0) / 2200.0, 0.0, 1.0) * (0.2 + 0.8 * voicing_arr)
    bright_20 = median_filter(bright_raw, size=20)

    loud_out = robust_norm(np.interp(t_grid, t_20, loud_20))
    voic_out = np.clip(np.interp(t_grid, t_20, voic_20), 0.0, 1.0).astype(np.float32)
    pitch_out = robust_norm(np.interp(t_grid, t_20, pitch_20))
    bright_out = robust_norm(np.interp(t_grid, t_20, bright_20))
    return loud_out, voic_out, pitch_out, bright_out


def compute_rhythm_dsp(audio: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """
    Low-mid rhythmic / periodic impact detector (80-1200 Hz):
      - 40 Hz log-energy envelope + positive onset flux
      - 4s sliding window autocorrelation in 0.8-5.0 Hz (48-300 cycles/min)
      - Combined with sustained energy floor & onset flux density
    """
    sos_r = butter(4, [80, 1200], btype="bandpass", fs=SR, output="sos")
    r_audio = sosfiltfilt(sos_r, audio).astype(np.float32)

    env_sr = 40  # 25 ms blocks
    block = SR // env_sr
    n_env = max(1, len(r_audio) // block)
    env = np.sqrt(np.mean(r_audio[: n_env * block].reshape(n_env, block) ** 2, axis=1) + 1e-10)
    env_log = np.log1p(env * 25.0).astype(np.float32)

    # Half-wave rectified energy flux (highlights repeating impacts)
    flux = np.maximum(0.0, np.diff(env_log, prepend=env_log[0])).astype(np.float32)

    win_n = 4 * env_sr  # 4-second window (160 samples @ 40 Hz)
    half = win_n // 2
    env_pad = np.pad(env_log, (half, half), mode="reflect")
    flux_pad = np.pad(flux, (half, half), mode="reflect")

    # Autocorrelation lag range: 0.20s (5 Hz / 300 BPM) to 1.25s (0.8 Hz / 48 BPM)
    lag_min, lag_max = 8, 50
    fft_n = 512

    n_out = len(t_grid)
    periodicity_arr = np.zeros(n_out, dtype=np.float32)
    vigor_arr = np.zeros(n_out, dtype=np.float32)

    for i, t in enumerate(t_grid):
        c = min(n_env - 1, int(t * env_sr))
        w_env = env_pad[c : c + win_n]
        w_flux = flux_pad[c : c + win_n]

        # Sustained floor (30th percentile kills isolated single impacts)
        sustained = float(np.percentile(w_env, 30))
        flux_mean = float(np.mean(w_flux))
        vigor_arr[i] = 0.45 * sustained + 0.55 * flux_mean

        # Normalized autocorrelation of onset flux
        ac = w_flux - np.mean(w_flux)
        if float(np.dot(ac, ac)) < 1e-8:
            continue
        f_ac = np.fft.rfft(ac, n=fft_n)
        acf = np.fft.irfft(np.abs(f_ac) ** 2, n=fft_n)[: lag_max + 2]
        acf_norm = acf / (acf[0] + 1e-10)

        mid = acf_norm[lag_min : lag_max + 1]
        left = acf_norm[lag_min - 1 : lag_max]
        right = acf_norm[lag_min + 1 : lag_max + 2]
        peaks = np.where((mid >= left) & (mid >= right) & (mid > 0.0), mid, 0.0)
        periodicity_arr[i] = float(np.max(peaks))

    vigor_norm = robust_norm(vigor_arr)
    period_norm = robust_norm(periodicity_arr)
    # Require BOTH rhythmic repetition and physical vigor (quiet ticking won't score high)
    rhythm_raw = vigor_norm * (0.25 + 0.75 * period_norm)
    return robust_norm(gaussian_filter1d(rhythm_raw, sigma=2.0))


def extract_peaks(score: np.ndarray, vocal: np.ndarray, rhythm: np.ndarray, max_peaks=20):
    peaks, _ = find_peaks(
        score,
        height=0.46,
        distance=25 * OUT_HZ,   # >= 25 seconds between distinct peak scenes
        prominence=0.14,
    )
    results = []
    n = len(score)
    for idx in peaks:
        val = float(score[idx])
        thresh = max(0.32, val * 0.58)
        s = int(idx)
        while s > 0 and score[s - 1] >= thresh:
            s -= 1
        e = int(idx)
        while e < n - 1 and score[e + 1] >= thresh:
            e += 1

        start_sec = max(0.0, round(s / OUT_HZ - 3.0, 1))  # 3s lead-in before rise
        end_sec = round(e / OUT_HZ, 1)
        results.append({
            "t": round(idx / OUT_HZ, 1),
            "start": start_sec,
            "end": end_sec,
            "score": round(val, 3),
            "vocal": round(float(np.mean(vocal[s : e + 1])), 3),
            "rhythm": round(float(np.mean(rhythm[s : e + 1])), 3),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:max_peaks]


def analyze_video(playback_id: str, interpreter: Interpreter, labels: list[str]) -> dict:
    audio = fetch_audio_pcm(playback_id)
    duration = len(audio) / SR
    n_pts = max(2, int(round(duration * OUT_HZ)))
    t_grid = (np.arange(n_pts, dtype=np.float32) + 0.5) / OUT_HZ

    yamnet_v = compute_yamnet_vocal(audio, interpreter, labels, t_grid)
    v_loud, voicing, pitch, bright = compute_vocal_dsp(audio, t_grid)
    rhythm = compute_rhythm_dsp(audio, t_grid)

    # Vocal score: YAMNet + voicing gate, scaled by loudness + pitch height + brightness
    vocal_gate = np.clip(0.65 * yamnet_v + 0.35 * voicing, 0.0, 1.0)
    vocal_strength = 0.45 * v_loud + 0.35 * pitch + 0.20 * bright
    vocal = robust_norm(gaussian_filter1d(vocal_gate * (0.30 + 0.70 * vocal_strength), sigma=1.5))

    # Synergy: strongly rewards simultaneous vigorous rhythm + intense vocalizations
    synergy = 0.35 * vocal + 0.25 * rhythm + 0.40 * (vocal * rhythm)
    synergy_med = median_filter(synergy, size=3)  # 1.5s median kills brief bumps

    # Continuity boost: reward stretches staying elevated across a 5-second window
    elevated = (synergy_med > np.percentile(synergy_med, 55)).astype(np.float32)
    continuity = uniform_filter1d(elevated, size=5 * OUT_HZ)
    score_raw = synergy_med * (0.65 + 0.35 * continuity)

    # Light smoothing (1.0s sigma) for clean scrubbing
    score = robust_norm(gaussian_filter1d(score_raw, sigma=2.0), low=2.0, high=98.0)

    peaks = extract_peaks(score, vocal, rhythm)

    return {
        "version": SCHEMA_VERSION,
        "playbackId": playback_id,
        "duration": round(duration, 2),
        "hz": OUT_HZ,
        "score": [int(round(v * 1000)) for v in score],
        "vocal": [int(round(v * 1000)) for v in vocal],
        "rhythm": [int(round(v * 1000)) for v in rhythm],
        "peaks": peaks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-run", type=int, default=15, help="Cap work per CI run")
    parser.add_argument("--force", action="store_true", help="Re-analyze existing files")
    parser.add_argument("--prune", action="store_true", help="Delete JSON for removed videos")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assets = json.loads(ASSETS_PATH.read_text(encoding="utf-8"))
    videos = assets.get("videos", [])
    valid_ids = {v["playbackId"] for v in videos if v.get("playbackId")}

    if args.prune:
        for p in OUT_DIR.glob("*.json"):
            if p.stem not in valid_ids:
                p.unlink()
                print(f"Pruned orphaned {p.name}")

    todo = [
        v for v in videos
        if v.get("playbackId") and (args.force or not (OUT_DIR / f"{v['playbackId']}.json").exists())
    ]

    if not todo:
        print("All videos already analyzed.")
        return

    model_path, labels = ensure_model()
    interpreter = Interpreter(model_path=str(model_path), num_threads=os.cpu_count() or 2)
    interpreter.allocate_tensors()

    processed = 0
    for v in todo[: args.max_per_run]:
        pid = v["playbackId"]
        print(f"[{processed + 1}/{min(len(todo), args.max_per_run)}] Analyzing {pid} ({v.get('title', '')})...")
        try:
            result = analyze_video(pid, interpreter, labels)
            out_path = OUT_DIR / f"{pid}.json"
            out_path.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
            processed += 1
        except Exception as e:
            print(f"  FAILED {pid}: {e}")

    print(f"Done. Processed {processed} videos ({max(0, len(todo) - processed)} remaining).")


if __name__ == "__main__":
    main()
