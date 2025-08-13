import os
import sys
import json
import shutil
import tempfile
import subprocess
from fractions import Fraction
from typing import Dict, List, Optional, Tuple


def get_sources_dir() -> str:
    default_dir = os.path.join(os.getcwd(), "sources")
    return os.environ.get("SOURCES_DIR", default_dir)


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def has_ffprobe() -> bool:
    return shutil.which("ffprobe") is not None


def list_video_files_by_mtime(directory: str) -> List[str]:
    video_exts = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".wmv", ".flv"}
    if not os.path.isdir(directory):
        return []
    entries: List[Tuple[float, str]] = []
    for name in os.listdir(directory):
        full = os.path.join(directory, name)
        if not os.path.isfile(full):
            continue
        if os.path.splitext(full)[1].lower() not in video_exts:
            continue
        try:
            mtime = os.path.getmtime(full)
        except OSError:
            continue
        entries.append((mtime, full))
    entries.sort(key=lambda x: x[0])
    return [p for _, p in entries]


def ffprobe_streams(path: str) -> Optional[Dict[str, object]]:
    if not has_ffprobe():
        return None
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        v_stream = None
        a_stream = None
        for s in data.get("streams", []):
            if s.get("codec_type") == "video" and v_stream is None:
                v_stream = s
            elif s.get("codec_type") == "audio" and a_stream is None:
                a_stream = s

        v_codec = v_stream.get("codec_name") if v_stream else None
        width = int(v_stream.get("width")) if v_stream and v_stream.get("width") else None
        height = int(v_stream.get("height")) if v_stream and v_stream.get("height") else None
        fr = None
        if v_stream is not None:
            fr = v_stream.get("avg_frame_rate") or v_stream.get("r_frame_rate")
        fps_value: Optional[float] = None
        if isinstance(fr, str) and fr not in ("0/0", "0"):
            try:
                fps_value = float(Fraction(fr))
            except Exception:
                fps_value = None

        a_codec = a_stream.get("codec_name") if a_stream else None

        return {
            "v_codec": v_codec,
            "a_codec": a_codec,
            "width": width,
            "height": height,
            "fps": fps_value,
        }
    except Exception:
        return None


def ffprobe_duration(path: str) -> Optional[float]:
    if not has_ffprobe():
        return None
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        fmt = data.get("format", {})
        dur = fmt.get("duration")
        if dur is None:
            return None
        try:
            return float(dur)
        except Exception:
            return None
    except Exception:
        return None


def ffprobe_keyframes(path: str) -> List[float]:
    """Return list of keyframe times (seconds) for the first video stream."""
    if not has_ffprobe():
        return []
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-skip_frame",
            "nokey",
            "-show_frames",
            "-print_format",
            "json",
            "-show_entries",
            "frame=pkt_pts_time",
            path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        frames = data.get("frames", [])
        times: List[float] = []
        for f in frames:
            t = f.get("pkt_pts_time")
            if t is None:
                continue
            try:
                times.append(float(t))
            except Exception:
                pass
        times.sort()
        return times
    except Exception:
        return []


def keyframe_at_or_before(keyframes: List[float], target: float) -> float:
    """Return the greatest keyframe time <= target, or 0.0 if none."""
    best = 0.0
    for t in keyframes:
        if t <= target and t >= best:
            best = t
    return best


def detect_peak_time(path: str, window_start: float, window_end: float) -> Optional[float]:
    """Scan mono float32 PCM samples via ffmpeg and return the time (s) of the
    largest absolute amplitude within [window_start, window_end].

    Uses 16 kHz mono to keep it fast. Returns absolute media time.
    """
    if window_end <= window_start:
        return None
    duration = window_end - window_start
    try:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{window_start:.6f}",
            "-t",
            f"{duration:.6f}",
            "-i",
            path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            "-",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        assert proc.stdout is not None
        sample_rate = 16000
        frame_bytes = 4  # float32
        chunk = proc.stdout.read()
        if not chunk:
            proc.wait()
            return None
        import struct
        count = len(chunk) // frame_bytes
        fmt = f"{count}f"
        samples = struct.unpack(fmt, chunk[:count * frame_bytes])
        # Scan for max absolute value; compute time index
        max_idx = 0
        max_val = 0.0
        for idx, v in enumerate(samples):
            av = v if v >= 0 else -v
            if av > max_val:
                max_val = av
                max_idx = idx
        rel_time = max_idx / float(sample_rate)
        return window_start + rel_time
    except Exception:
        return None


def ensure_same_params(meta_a: Dict[str, object], meta_b: Dict[str, object]) -> Tuple[bool, str]:
    def nearly_equal(x: Optional[float], y: Optional[float], tol: float = 1e-3) -> bool:
        if x is None or y is None:
            return x == y
        return abs(x - y) <= tol

    checks = [
        (meta_a.get("v_codec"), meta_b.get("v_codec"), "video codec"),
        (meta_a.get("a_codec"), meta_b.get("a_codec"), "audio codec"),
        (meta_a.get("width"), meta_b.get("width"), "width"),
        (meta_a.get("height"), meta_b.get("height"), "height"),
    ]
    for lhs, rhs, label in checks:
        if lhs != rhs:
            return False, f"Mismatch in {label}: {lhs} vs {rhs}"

    if not nearly_equal(meta_a.get("fps"), meta_b.get("fps")):
        return False, f"Mismatch in fps: {meta_a.get('fps')} vs {meta_b.get('fps')}"

    return True, "ok"


def write_concat_list_file(paths: List[str]) -> str:
    # Use forward slashes to avoid Windows escaping issues
    lines = []
    for p in paths:
        norm = p.replace("\\", "/")
        # Escape single quotes if present
        norm = norm.replace("'", "'\\''")
        lines.append(f"file '{norm}'\n")
    fd, list_path = tempfile.mkstemp(prefix="concat_", suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return list_path


def write_ffconcat(entries: List[Dict[str, object]]) -> str:
    """Write an ffconcat list with optional inpoint/outpoint per file.

    entries: [{"file": path, "in": float|None, "out": float|None}]
    Returns the filepath of the temp list.
    """
    lines = ["ffconcat version 1.0\n"]
    for e in entries:
        path_abs = os.path.abspath(str(e["file"]))
        path = path_abs.replace("\\", "/").replace("'", "'\\''")
        lines.append(f"file '{path}'\n")
        if e.get("in") is not None:
            lines.append(f"inpoint {float(e['in']):.6f}\n")
        if e.get("out") is not None:
            lines.append(f"outpoint {float(e['out']):.6f}\n")
    fd, list_path = tempfile.mkstemp(prefix="ffconcat_", suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return list_path


def concat_videos_copy(paths: List[str], output_path: str) -> int:
    list_file = write_concat_list_file(paths)
    try:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file,
            "-c",
            "copy",
            output_path,
        ]
        proc = subprocess.run(cmd)
        return proc.returncode
    finally:
        try:
            os.remove(list_file)
        except OSError:
            pass


def concat_videos_copy_video_only(paths: List[str], output_path: str) -> int:
    list_file = write_concat_list_file(paths)
    try:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file,
            "-an",
            "-c",
            "copy",
            output_path,
        ]
        proc = subprocess.run(cmd)
        return proc.returncode
    finally:
        try:
            os.remove(list_file)
        except OSError:
            pass


def concat_video_j_copy(a_path: str, a_out: float, b_path: str, b_in_adj: float, output_path: str) -> int:
    # J-cut video: A ends at a_out, B starts at adjusted inpoint (keyframe-aligned)
    ov = max(0.0, b_in_adj)
    entries = [
        {"file": a_path, "out": a_out},
        {"file": b_path, "in": ov},
    ]
    lst = write_ffconcat(entries)
    try:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            lst,
            "-an",
            "-c",
            "copy",
            output_path,
        ]
        return subprocess.run(cmd).returncode
    finally:
        try:
            os.remove(lst)
        except OSError:
            pass


def concat_video_l_copy(a_path: str, a_out_adj: float, b_path: str, output_path: str) -> int:
    # L-cut video: end A at adjusted outpoint, then full B
    cut_a = max(0.0, a_out_adj)
    entries = [
        {"file": a_path, "out": cut_a},
        {"file": b_path},
    ]
    lst = write_ffconcat(entries)
    try:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            lst,
            "-an",
            "-c",
            "copy",
            output_path,
        ]
        return subprocess.run(cmd).returncode
    finally:
        try:
            os.remove(lst)
        except OSError:
            pass


def build_audio_j_curve(input_a: str, input_b: str, output_audio: str, dur_a: float, dur_b: float, b_inpoint: float, overlap: float) -> int:
    """J-curve com cortes exatos:
    - Vídeo B começa em b_inpoint; vídeo A vai até dur_a.
    - Áudio B antes do corte: segmento [b_inpoint-overlap, b_inpoint] com fade-in.
    - Áudio B após o corte: segmento [b_inpoint, fim] começando exatamente no corte.
    - Áudio A: fade-out nos últimos 'overlap' segundos.
    """
    cut_time = dur_a
    pre_start = max(b_inpoint - overlap, 0.0)
    pre_len = b_inpoint - pre_start
    total = dur_a + dur_b - overlap
    ms_pre = int(max(cut_time - pre_len, 0.0) * 1000)
    ms_post = int(cut_time * 1000)

    filter_graph = (
        f"[0:a]atrim=0:{dur_a:.6f},asetpts=PTS-STARTPTS,afade=t=out:st={dur_a-overlap:.6f}:d={overlap:.6f}[a0];"
        f"[1:a]atrim={pre_start:.6f}:{b_inpoint:.6f},asetpts=PTS-STARTPTS,afade=t=in:st=0:d={pre_len:.6f}[be];"
        f"[be]adelay={ms_pre}:all=1[be_d];"
        f"[1:a]atrim={b_inpoint:.6f}:{dur_b:.6f},asetpts=PTS-STARTPTS[bp];"
        f"[bp]adelay={ms_post}:all=1[bp_d];"
        f"[a0][be_d][bp_d]amix=inputs=3:normalize=0:duration=longest[a];"
        f"[a]apad[a2]"
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        input_a,
        "-i",
        input_b,
        "-filter_complex",
        filter_graph,
        "-map",
        "[a2]",
        "-t",
        f"{total:.6f}",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        output_audio,
    ]
    proc = subprocess.run(cmd)
    return proc.returncode


def build_audio_l_curve(input_a: str, input_b: str, output_audio: str, dur_a: float, dur_b: float, cut_time: float, overlap: float) -> int:
    """L-curve com cortes exatos:
    - Vídeo A termina em cut_time; vídeo B começa em 0.
    - Áudio A: fade-out nos últimos 'overlap' segundos (terminando em dur_a).
    - Áudio B: começa no corte (delay cut_time) com fade-in de 'overlap'.
    """
    ms_b = int(max(cut_time, 0.0) * 1000)
    total = cut_time + dur_b
    filter_graph = (
        f"[0:a]atrim=0:{dur_a:.6f},asetpts=PTS-STARTPTS,afade=t=out:st={dur_a-overlap:.6f}:d={overlap:.6f}[a0];"
        f"[1:a]atrim=0:{dur_b:.6f},asetpts=PTS-STARTPTS,afade=t=in:st=0:d={overlap:.6f}[b0];"
        f"[b0]adelay={ms_b}:all=1[b0_d];"
        f"[a0][b0_d]amix=inputs=2:normalize=0:duration=longest[a];"
        f"[a]apad[a2]"
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        input_a,
        "-i",
        input_b,
        "-filter_complex",
        filter_graph,
        "-map",
        "[a2]",
        "-t",
        f"{total:.6f}",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        output_audio,
    ]
    proc = subprocess.run(cmd)
    return proc.returncode


def pad_audio_to_duration(input_audio: str, target_duration: float, output_audio: str) -> int:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        input_audio,
        "-filter_complex",
        "apad",
        "-t",
        f"{target_duration:.3f}",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        output_audio,
    ]
    proc = subprocess.run(cmd)
    return proc.returncode


def mux_video_and_audio(video_path: str, audio_path: str, output_path: str) -> int:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        output_path,
    ]
    proc = subprocess.run(cmd)
    return proc.returncode


def safe_remove(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def merge_two_j(a_path: str, b_path: str, overlap: float, out_path: str) -> int:
    dur_a = ffprobe_duration(a_path) or 0.0
    dur_b = ffprobe_duration(b_path) or 0.0

    # 1) Determine cut time using audio peak in last 3s but > 1s from end
    win_start = max(dur_a - 3.0, 0.0)
    win_end = max(dur_a - 1.0, 0.0)
    peak = detect_peak_time(a_path, win_start, win_end)
    cut_time_raw = peak if peak is not None else max(dur_a - overlap, 0.0)

    # 2) Align A outpoint to nearest keyframe at or before cut
    kf_a = ffprobe_keyframes(a_path)
    a_out_adj = keyframe_at_or_before(kf_a, cut_time_raw) if kf_a else cut_time_raw

    # 3) Compute B inpoint aligned to keyframe (aim for 1s prelap)
    kf_b = ffprobe_keyframes(b_path)
    b_in_adj = keyframe_at_or_before(kf_b, overlap) if kf_b else overlap

    # Temp products
    outputs_dir = os.path.dirname(out_path) or os.getcwd()
    tmp_video = os.path.join(outputs_dir, "tmp_step_video.mp4")
    tmp_audio = os.path.join(outputs_dir, "tmp_step_audio.m4a")

    safe_remove(tmp_video)
    safe_remove(tmp_audio)

    # Build video (copy only) J-cut with dynamic A outpoint
    code = concat_video_j_copy(a_path, a_out_adj, b_path, b_in_adj, tmp_video)
    if code != 0:
        return code

    # Build audio aligned to trims using cut_time anchored to a_out_adj
    code = build_audio_j_curve(a_path, b_path, tmp_audio, a_out_adj, dur_b, b_inpoint=b_in_adj, overlap=overlap)
    if code != 0:
        return code

    # Mux into final
    code = mux_video_and_audio(tmp_video, tmp_audio, out_path)
    if code != 0:
        return code

    safe_remove(tmp_video)
    safe_remove(tmp_audio)
    return 0


def main(argv: List[str]) -> int:
    if not has_ffmpeg():
        print("Erro: ffmpeg não encontrado no PATH. Instale FFmpeg para continuar.", file=sys.stderr)
        return 2
    if not has_ffprobe():
        print("Erro: ffprobe não encontrado no PATH. Instale FFmpeg (inclui ffprobe).", file=sys.stderr)
        return 2

    sources_dir = argv[1] if len(argv) > 1 else get_sources_dir()
    files = list_video_files_by_mtime(sources_dir)
    if len(files) < 3:
        print("Não há vídeos suficientes em 'sources/' para o teste (necessário ao menos 3).", file=sys.stderr)
        return 1

    # Validate codecs and fps match across three
    metas = [ffprobe_streams(p) for p in files[:3]]
    if any(m is None for m in metas):
        print("Falha ao inspecionar metadados com ffprobe.", file=sys.stderr)
        return 2
    ok, reason = ensure_same_params(metas[0], metas[1])
    if not ok:
        print(f"Parâmetros incompatíveis entre vídeo 1 e 2: {reason}", file=sys.stderr)
        return 3
    ok, reason = ensure_same_params(metas[1], metas[2])
    if not ok:
        print(f"Parâmetros incompatíveis entre vídeo 2 e 3: {reason}", file=sys.stderr)
        return 3

    outputs_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    overlap = 1.0
    # Step 1: merge 1 and 2 into temp
    tmp_step1 = os.path.join(outputs_dir, "tmp_step1_j.mp4")
    safe_remove(tmp_step1)
    print("Mesclando (J) vídeos 1 e 2...")
    code = merge_two_j(files[0], files[1], overlap=overlap, out_path=tmp_step1)
    if code != 0:
        return code

    # Step 2: merge result with 3 → final 003
    final_out = os.path.join(outputs_dir, "outputs_003_merged_first_three_J.mp4")
    safe_remove(final_out)
    print("Mesclando (J) resultado com vídeo 3 -> 003...")
    code = merge_two_j(tmp_step1, files[2], overlap=overlap, out_path=final_out)
    if code != 0:
        return code

    safe_remove(tmp_step1)
    print(f"Concluído com sucesso. Saída: {final_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


