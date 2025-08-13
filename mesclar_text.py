import os
import sys
import tempfile
import shutil
import subprocess
from typing import List, Optional, Dict

from mesclar import (
    ffprobe_duration,
    ffprobe_keyframes,
    keyframe_at_or_before,
    concat_video_j_copy,
    build_audio_j_curve,
    mux_video_and_audio,
    safe_remove,
    get_sources_dir,
)


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def extract_audio_segment(src_video: str, start: float, duration: float, out_wav: str, sample_rate: int = 16000) -> int:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        src_video,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        out_wav,
    ]
    return subprocess.run(cmd).returncode


def transcribe_with_whisper(audio_path: str, language: str = "pt", model_name: str = "base") -> Optional[List[Dict[str, object]]]:
    try:
        import whisper  # type: ignore
    except Exception:
        return None
    try:
        model = whisper.load_model(model_name)
        result = model.transcribe(audio_path, language=language, word_timestamps=True, fp16=False)
        segments = result.get("segments", [])
        return segments  # each: {start, end, text}
    except Exception:
        return None


def normalize_text(t: str) -> str:
    return "".join(ch.lower() for ch in t if ch.isalnum() or ch.isspace())


def find_phrase_time_in_segments(segments: List[Dict[str, object]], phrase: str) -> Optional[float]:
    if not segments:
        return None
    phrase_norm = normalize_text(phrase)
    # Search from the end towards the start
    for seg in reversed(segments):
        text = normalize_text(str(seg.get("text", "")))
        if phrase_norm in text:
            try:
                return float(seg.get("start", 0.0))
            except Exception:
                return None
    return None


def find_last_word_times(segments: List[Dict[str, object]], window_start: float, window_end: float, total_duration: float) -> Optional[Dict[str, float]]:
    """Return dict with 'start' and 'end' (relative to window) for the last word ending inside window.

    Falls back to segment end if word timings unavailable.
    """
    if not segments:
        return None
    best_rel_end: float = -1.0
    best_rel_start: float = -1.0
    # Iterate all segments; collect word timings if present
    for seg in segments:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", 0.0))
        # Words may be available
        words = seg.get("words") or []
        if words:
            for w in words:
                try:
                    ws = float(w.get("start", seg_start))
                    we = float(w.get("end", seg_end))
                except Exception:
                    continue
                rel_start = ws
                rel_end = we
                # We are already in window-relative coordinates as we transcribed the window clip
                if rel_end >= 0.0 and rel_end <= (window_end - window_start):
                    if rel_end > best_rel_end:
                        best_rel_end = rel_end
                        best_rel_start = rel_start
        else:
            # Fallback: use segment end
            rel_end = seg_end
            rel_start = seg_start
            if rel_end >= 0.0 and rel_end <= (window_end - window_start):
                if rel_end > best_rel_end:
                    best_rel_end = rel_end
                    best_rel_start = rel_start

    if best_rel_end >= 0.0:
        return {"start": best_rel_start, "end": best_rel_end}
    return None


def find_closing_phrase_time(segments: List[Dict[str, object]], window_start: float, window_end: float, total_duration: float) -> Optional[float]:
    """Pick the end time of the last spoken segment inside the window.

    If no segment ends inside [window_start, window_end], pick the latest
    segment end before total_duration - 1s.
    """
    if not segments:
        return None
    best: Optional[float] = None
    for seg in segments:
        try:
            s = float(seg.get("start", 0.0))
            e = float(seg.get("end", 0.0))
        except Exception:
            continue
        if e >= window_start and e <= window_end:
            if best is None or e > best:
                best = e
    if best is not None:
        return best
    cutoff = max(total_duration - 1.0, 0.0)
    for seg in reversed(segments):
        try:
            e = float(seg.get("end", 0.0))
        except Exception:
            continue
        if e <= cutoff:
            return e
    return None


def merge_two_j_text(a_path: str, b_path: str, phrase: Optional[str] = None, overlap: float = 1.0, window_seconds: float = 5.0, exclude_tail_seconds: float = 1.0, output_path: Optional[str] = None) -> int:
    if not has_ffmpeg():
        print("ffmpeg não encontrado no PATH.", file=sys.stderr)
        return 2

    dur_a = ffprobe_duration(a_path) or 0.0
    dur_b = ffprobe_duration(b_path) or 0.0

    win_start = max(dur_a - window_seconds, 0.0)
    win_end = max(dur_a - exclude_tail_seconds, 0.0)
    win_dur = max(0.0, win_end - win_start)

    # Extract small audio clip to a temp wav for ASR
    fd, wav_path = tempfile.mkstemp(prefix="valeu_seg_", suffix=".wav")
    os.close(fd)
    try:
        if win_dur > 0 and extract_audio_segment(a_path, win_start, win_dur, wav_path) == 0:
            segments = transcribe_with_whisper(wav_path)
        else:
            segments = None
    finally:
        safe_remove(wav_path)

    cut_time_raw: Optional[float] = None
    dynamic_overlap: Optional[float] = None
    if segments:
        if phrase:
            rel_time = find_phrase_time_in_segments(segments, phrase=phrase)
            if rel_time is not None:
                cut_time_raw = win_start + rel_time
        if cut_time_raw is None:
            # Try last word end inside window
            lw = find_last_word_times(segments, win_start, win_end, total_duration=win_dur)
            if lw is not None:
                cut_time_raw = win_start + lw["end"]
                dynamic_overlap = max(lw["end"] - lw["start"], 0.0)
        if cut_time_raw is None:
            rel_time2 = find_closing_phrase_time(segments, 0.0, win_dur, total_duration=win_dur)
            if rel_time2 is not None:
                cut_time_raw = win_start + rel_time2

    if cut_time_raw is None:
        # Fallback: default J-cut of 1s
        cut_time_raw = max(dur_a - overlap, 0.0)
        dynamic_overlap = overlap

    # Constrain overlap to a reasonable range
    if dynamic_overlap is None:
        dynamic_overlap = overlap
    dynamic_overlap = max(0.1, min(dynamic_overlap, 2.0))

    # Align video to keyframes
    kf_a = ffprobe_keyframes(a_path)
    a_out_adj = keyframe_at_or_before(kf_a, cut_time_raw) if kf_a else cut_time_raw

    kf_b = ffprobe_keyframes(b_path)
    b_in_adj = keyframe_at_or_before(kf_b, dynamic_overlap) if kf_b else dynamic_overlap

    # Build temp outputs
    outputs_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    tmp_video = os.path.join(outputs_dir, "tmp_text_video.mp4")
    tmp_audio = os.path.join(outputs_dir, "tmp_text_audio.m4a")
    safe_remove(tmp_video)
    safe_remove(tmp_audio)

    code = concat_video_j_copy(a_path, a_out_adj, b_path, b_in_adj, tmp_video)
    if code != 0:
        return code

    code = build_audio_j_curve(a_path, b_path, tmp_audio, a_out_adj, dur_b, b_inpoint=b_in_adj, overlap=dynamic_overlap)
    if code != 0:
        return code

    if output_path is None:
        output_path = os.path.join(outputs_dir, "outputs_004_text_J.mp4")
    safe_remove(output_path)
    code = mux_video_and_audio(tmp_video, tmp_audio, output_path)
    if code != 0:
        return code

    safe_remove(tmp_video)
    safe_remove(tmp_audio)
    print(f"Concluído. Saída: {output_path}")
    return 0


def main(argv: List[str]) -> int:
    sources_dir = get_sources_dir()
    # Teste solicitado: vídeos 3 e 4 -> GX015501 + GX015502
    a = os.path.join(sources_dir, "GX015501.MP4")
    b = os.path.join(sources_dir, "GX015502.MP4")
    out = os.path.join(os.getcwd(), "outputs", "outputs_006_text_J_501_502.mp4")
    # Usar janela final de 3s e excluir último 1s, procurando a última palavra (sem fixar "valeu")
    return merge_two_j_text(a, b, phrase=None, overlap=1.0, window_seconds=3.0, exclude_tail_seconds=1.0, output_path=out)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


