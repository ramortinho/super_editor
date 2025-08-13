import os
import sys
import math
import shutil
import tempfile
import subprocess
from typing import List, Tuple, Optional, Dict

from cortar import transcribe_full  # Reutiliza Whisper do projeto
from mesclar import ffprobe_duration  # Utilitários já existentes


MUSIC_PATH_DEFAULT = os.path.join(os.getcwd(), "sounds", "03.04 - Introdução e Shorts (Remix).mp3")
MERGED_VIDEO_DEFAULT = os.path.join(os.getcwd(), "outputs", "final_merged_J_text.mp4")


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def read_audio_mono_f32(path: str, start: float = 0.0, duration: Optional[float] = None, sample_rate: int = 16000) -> Tuple[List[float], int]:
    """Read audio as float32 mono via ffmpeg. Returns samples list and sample_rate."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, start):.6f}",
    ]
    if duration is not None and duration > 0:
        cmd += ["-t", f"{duration:.6f}"]
    cmd += [
        "-i",
        path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        return [], sample_rate
    raw = proc.stdout
    # Interpret bytes as float32 little-endian
    import struct
    count = len(raw) // 4
    if count <= 0:
        return [], sample_rate
    fmt = f"{count}f"
    samples = struct.unpack(fmt, raw[: count * 4])
    return list(samples), sample_rate


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1 or not values:
        return values[:]
    acc = [0.0]
    for v in values:
        acc.append(acc[-1] + abs(v))
    out: List[float] = []
    for i in range(len(values)):
        j = min(len(values), i + window)
        i0 = max(0, i - window + 1)
        out.append((acc[j] - acc[i0]) / (j - i0))
    return out


def detect_beats(path: str, sample_rate: int = 16000, smooth_ms: int = 100, min_spacing_ms: int = 450, peak_quantile: float = 0.8) -> List[float]:
    """Very light-weight beat/onset detection using short-time energy peaks."""
    samples, sr = read_audio_mono_f32(path, sample_rate=sample_rate)
    if not samples:
        return []
    win = max(1, int((smooth_ms / 1000.0) * sr))
    env = moving_average(samples, win)
    # Threshold by quantile
    if not env:
        return []
    sorted_env = sorted(env)
    q_idx = int(max(0, min(len(sorted_env) - 1, peak_quantile * (len(sorted_env) - 1))))
    thr = sorted_env[q_idx]

    beats: List[float] = []
    min_gap = min_spacing_ms / 1000.0
    last_t = -1e9
    for i in range(1, len(env) - 1):
        if env[i] > thr and env[i] >= env[i - 1] and env[i] >= env[i + 1]:
            t = i / sr
            if t - last_t >= min_gap:
                beats.append(t)
                last_t = t
    return beats


def choose_speech_segments_by_energy(video_path: str, window_pad: float = 0.0, min_len: float = 0.8, max_len: float = 1.8, top_k: int = 12) -> List[Tuple[float, float]]:
    """Transcribe and select energetic short segments suitable for intro."""
    segs = transcribe_full(video_path, language="pt", model_name="base")
    if not segs:
        return []
    candidates: List[Tuple[float, float, float]] = []  # (score, start, end)
    for seg in segs:
        try:
            s = float(seg.get("start", 0.0))
            e = float(seg.get("end", 0.0))
        except Exception:
            continue
        dur = e - s
        if dur < min_len or dur > max_len:
            continue
        # Compute RMS energy as score
        samples, sr = read_audio_mono_f32(video_path, start=max(0.0, s - window_pad), duration=dur + 2 * window_pad)
        if not samples:
            continue
        import math as _m
        rms = _m.sqrt(sum(v * v for v in samples) / max(1, len(samples)))
        candidates.append((rms, s, e))

    candidates.sort(reverse=True, key=lambda x: x[0])
    pick = [(s, e) for _, s, e in candidates[:top_k]]
    return pick


def extract_audio_segment(src: str, start: float, duration: float, out_path: str) -> int:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        src,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "48000",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        out_path,
    ]
    return subprocess.run(cmd).returncode


def mix_intro(music_path: str, speech_segments: List[Tuple[float, float]], speech_source: str, beats: List[float], out_path: str, music_db_reduction: float = 5.0) -> int:
    """Place selected speech segments on nearest beats and mix over music with -5dB volume reduction."""
    if not beats or not speech_segments:
        return 1
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Pair segments with beats (round-robin / zip)
    n = min(len(beats), len(speech_segments))
    pairs = list(zip(beats[:n], speech_segments[:n]))

    # Prepare temp speech files
    tmp_files: List[str] = []
    try:
        for _, (s, e) in pairs:
            dur = max(0.1, e - s)
            fd, tmp = tempfile.mkstemp(prefix="intro_seg_", suffix=".m4a")
            os.close(fd)
            rc = extract_audio_segment(speech_source, s, dur, tmp)
            if rc != 0:
                continue
            tmp_files.append(tmp)

        # Build filter_complex: music volume down, adelay per speech, then amix
        # Inputs: 0 -> music, 1..N -> speech files
        inputs = ["-i", music_path]
        for f in tmp_files:
            inputs += ["-i", f]

        vol_linear = 10 ** (-(music_db_reduction) / 20.0)
        filter_parts: List[str] = [f"[0:a]volume={vol_linear:.6f}[bg]"]
        labels = ["[bg]"]
        for idx, (beat_time, _) in enumerate(pairs, start=1):
            delay_ms = int(max(0.0, beat_time) * 1000)
            filter_parts.append(f"[{idx}:a]adelay={delay_ms}:all=1[a{idx}d]")
            labels.append(f"[a{idx}d]")

        amix = "".join(labels)
        amix += f"amix=inputs={len(labels)}:normalize=0:duration=first[outa]"
        filter_complex = ";".join(filter_parts + [amix])

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
        ] + inputs + [
            "-filter_complex",
            filter_complex,
            "-map",
            "[outa]",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            out_path,
        ]
        return subprocess.run(cmd).returncode
    finally:
        for f in tmp_files:
            try:
                os.remove(f)
            except OSError:
                pass


def main(argv: List[str]) -> int:
    if not has_ffmpeg():
        print("ffmpeg não encontrado no PATH.", file=sys.stderr)
        return 2

    music_path = MUSIC_PATH_DEFAULT
    speech_source = MERGED_VIDEO_DEFAULT
    if len(argv) > 1:
        music_path = argv[1]
    if len(argv) > 2:
        speech_source = argv[2]

    if not os.path.exists(music_path):
        print(f"[erro] Música não encontrada: {music_path}", file=sys.stderr)
        return 1
    if not os.path.exists(speech_source):
        print(f"[erro] Fonte de falas não encontrada: {speech_source}", file=sys.stderr)
        return 1

    print("[info] Detectando batidas na música...")
    beats = detect_beats(music_path, sample_rate=16000, smooth_ms=100, min_spacing_ms=450, peak_quantile=0.85)
    print(f"[info] Batidas detectadas: {len(beats)}")

    print("[info] Selecionando trechos de fala energéticos...")
    segs = choose_speech_segments_by_energy(speech_source, window_pad=0.05, min_len=0.8, max_len=1.8, top_k=16)
    print(f"[info] Trechos de fala selecionados: {len(segs)}")

    out_path = os.path.join(os.getcwd(), "outputs", "intro_audio_mix.m4a")
    print("[info] Mixando intro...")
    rc = mix_intro(music_path, segs, speech_source, beats, out_path, music_db_reduction=5.0)
    if rc != 0:
        print(f"[erro] ffmpeg retornou código {rc}", file=sys.stderr)
        return rc
    print(f"[done] Intro gerada: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


