#!/usr/bin/env python3
"""
Incremental Option D audio intensity analyzer for Mux assets.
Reads public/assets.json, skips already-processed videos, and writes
public/intensity/<playbackId>.json.
"""
import argparse
import json
import os
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import butter, find_peaks, sosfiltfilt

ROOT = Path(__file__).resolve().parent.parent
ASSETS_PATH = ROOT / "public" / "assets.json"
OUT_DIR = ROOT / "public" / "intensity"
MODEL_DIR = ROOT / ".cache" / "yamnet"

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/latest/yamnet.tflite"
LABELS_URL = "https://storage.googleapis.com/mediapipe-tasks/audio_classifier/yamnet_label_list.txt"

SR = 16000
SCHEMA_VERSION = 1

# Tiered weights for AudioSet/YAMNet classes:
# Strong non-speech vocal reactions rank highest; calm speech ranks lower;
# background noise/music/rustling is penalized.
VOCAL_WEIGHTS = {
    # High-energy / intense vocalizations
    "Shout": 1.0, "Yell": 1.0, "Bellow": 1.0, "Whoop": 1.0, "Screaming": 1.0,
    "Cheering": 1.0, "Groan": 1.0, "Grunt": 1.0, "Gasp": 1.0, "Pant": 1.0,
    "Wail, moan": 1.0, "Whimper": 0.9, "Laughter": 0.85, "Giggle": 0.8,
    "Crying, sobbing": 0.8,
    # Moderate vocal/breathing cues
    "Breathing": 0.65, "Sigh": 0.6, "Snort": 0.55, "Chuckle, chortle": 0.6,
    # Regular talking (kept lower so calm commentary doesn't overshadow peaks)
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
        return x
    lo, hi = np.percentile(x, [low, high])
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def fetch_audio_pcm(playback_id: str) -> np.ndarray:
    """Stream 16kHz mono float32 PCM from Mux audio-only endpoint (with HLS fallback)."""
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


def compute_yamnet_vocal(audio: np.ndarray, interpreter: Interpreter, labels: list[str], n_sec: int) -> np.ndarray:
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    win_len = int(inp["shape"][-1])  # 15600 samples (0.975s) for MediaPipe YAMNet
    hop_len = SR // 2                # 0.5s hop (2 Hz)

    # Precompute weight vectors across the 521 classes
    w_vocal = np.zeros(len(labels), dtype=np.float32)
    w_penalty = np.zeros(len(labels), dtype=np.float32)
    for idx, name in enumerate(labels):
        if name in VOCAL_WEIGHTS:
            w_vocal[idx] = VOCAL_WEIGHTS[name]
        if name in PENALTY_CLASSES:
            w_penalty[idx] = 1.0

    times = []
    raw_scores = []

    for start in range(0, len(audio), hop_len):
        chunk = audio[start : start + win_len]
        if len(chunk) < win_len:
            chunk = np.pad(chunk, (0, win_len - len(chunk)))
        tensor = chunk.reshape(inp["shape"]).astype(np.float32)
        interpreter.set_tensor(inp["index"], tensor)
        interpreter.invoke()
        probs = interpreter.get_tensor(out["index"]).flatten()

        # Combine strongest vocal cue with total vocal probability; dampen music/noise
        vocal_peak = float(np.max(probs * w_vocal))
        vocal_sum = float(np.clip(np.sum(probs * w_vocal), 0.0, 1.5))
        noise_pen = float(np.clip(np.sum(probs * w_penalty), 0.0, 1.0))

        v = (0.65 * vocal_peak + 0.35 * min(vocal_sum, 1.0)) * (1.0 - 0.65 * noise_pen)
        times.append((start + win_len * 0.5) / SR)
        raw_scores.append(v)

    t_grid = np.arange(n_sec, dtype=np.float32) + 0.5
    vocal_1hz = np.interp(t_grid, times, raw_scores).astype(np.float32)
    # 3-second rolling median rejects single-frame misclassifications
    return median_filter(vocal_1hz, size=3)


def compute_dsp_features(audio: np.ndarray, n_sec: int):
    # 1. Sustained vocal-band loudness (200 - 3500 Hz)
    sos_v = butter(4, [200, 3500], btype="bandpass", fs=SR, output="sos")
    vocal_band = sosfiltfilt(sos_v, audio).astype(np.float32)

    # Compute 20 Hz RMS -> log-compress -> 3s rolling median (kills isolated bangs)
    step_20hz = SR // 20
    n_frames = len(vocal_band) // step_20hz
    rms_20hz = np.sqrt(
        np.mean(vocal_band[: n_frames * step_20hz].reshape(n_frames, step_20hz) ** 2, axis=1) + 1e-10
    )
    sustained_loud_20hz = median_filter(np.log1p(rms_20hz * 25.0), size=60)  # 3s median
    t_20 = (np.arange(n_frames) + 0.5) / 20.0
    t_1 = np.arange(n_sec, dtype=np.float32) + 0.5
    vocal_loud_1hz = robust_norm(np.interp(t_1, t_20, sustained_loud_20hz))

    # 2. Rhythmic activity via 1.0 - 5.0 Hz envelope modulation (60 - 300 BPM)
    sos_r = butter(4, [150, 2000], btype="bandpass", fs=SR, output="sos")
    rhythm_band = sosfiltfilt(sos_r, audio).astype(np.float32)

    env_sr = 40  # 40 Hz amplitude envelope (25ms blocks)
    block = SR // env_sr
    n_env = len(rhythm_band) // block
    env = np.sqrt(
        np.mean(rhythm_band[: n_env * block].reshape(n_env, block) ** 2, axis=1) + 1e-10
    )
    env_log = np.log1p(env * 25.0)

    win_sec = 6
    win_n = win_sec * env_sr
    hann = np.hanning(win_n)
    freqs = np.fft.rfftfreq(win_n, d=1.0 / env_sr)
    mod_mask = (freqs >= 1.0) & (freqs <= 5.0)

    rhythm_raw = np.zeros(n_sec, dtype=np.float32)
    half = win_n // 2
    padded = np.pad(env_log, (half, half), mode="reflect")

    for sec in range(n_sec):
        center = min(len(env_log) - 1, int((sec + 0.5) * env_sr))
        w = padded[center : center + win_n]
        # 35th percentile measures sustained activity floor (ignores single bumps)
        sustained_floor = float(np.percentile(w, 35))
        ac = (w - np.mean(w)) * hann
        spec = np.abs(np.fft.rfft(ac)) ** 2
        total_p = float(np.sum(spec[1:]) + 1e-10)
        mod_spec = spec[mod_mask]
        # Ratio of 1-5 Hz rhythmic energy + peak sharpness (tonality of rhythm)
        band_ratio = float(np.sum(mod_spec)) / total_p
        peak_sharpness = float(np.max(mod_spec)) / (float(np.mean(mod_spec)) + 1e-10)
        periodicity = band_ratio * min(peak_sharpness / 6.0, 1.5)
        rhythm_raw[sec] = sustained_floor * periodicity

    rhythm_1hz = robust_norm(gaussian_filter1d(rhythm_raw, sigma=2.0))
    return vocal_loud_1hz, rhythm_1hz


def extract_peaks(score: np.ndarray, vocal: np.ndarray, rhythm: np.ndarray, max_peaks=20):
    peaks, _ = find_peaks(score, height=0.48, distance=30, prominence=0.14)
    results = []
    n = len(score)
    for idx in peaks:
        val = float(score[idx])
        thresh = max(0.32, val * 0.60)
        s = int(idx)
        while s > 0 and score[s - 1] >= thresh:
            s -= 1
        e = int(idx)
        while e < n - 1 and score[e + 1] >= thresh:
            e += 1
        results.append({
            "t": int(idx),
            "start": max(0, s - 3),   # 3s lead-in before rise
            "end": int(e),
            "score": round(val, 3),
            "vocal": round(float(np.mean(vocal[s : e + 1])), 3),
            "rhythm": round(float(np.mean(rhythm[s : e + 1])), 3),
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:max_peaks]


def analyze_video(playback_id: str, interpreter: Interpreter, labels: list[str]) -> dict:
    audio = fetch_audio_pcm(playback_id)
    duration = len(audio) / SR
    n_sec = max(1, int(round(duration)))

    vocal_prob = compute_yamnet_vocal(audio, interpreter, labels, n_sec)
    vocal_loud, rhythm = compute_dsp_features(audio, n_sec)

    # Combine YAMNet vocal confidence with sustained vocal-band loudness
    vocal = robust_norm(gaussian_filter1d(vocal_prob * (0.35 + 0.65 * vocal_loud), sigma=1.5))

    # Synergy score: rewards simultaneous rhythmic vigor + strong vocal reactions
    combined_raw = 0.35 * vocal + 0.25 * rhythm + 0.40 * (vocal * rhythm)
    score = robust_norm(gaussian_filter1d(combined_raw, sigma=2.5), low=2.0, high=98.0)

    peaks = extract_peaks(score, vocal, rhythm)

    # Quantize to 0..1000 integers to keep JSON files compact (~20KB per hour)
    return {
        "version": SCHEMA_VERSION,
        "playbackId": playback_id,
        "duration": round(duration, 2),
        "hz": 1,
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