import os
import sys
import shutil
import subprocess
from typing import List, Dict, Optional, Tuple

from mesclar import (
    ffprobe_duration,
    ffprobe_keyframes,
    keyframe_at_or_before,
    write_ffconcat,
    pad_audio_to_duration,
    mux_video_and_audio,
    get_sources_dir,
)


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def transcribe_full(audio_video_path: str, language: str = "pt", model_name: str = "base") -> Optional[List[Dict[str, object]]]:
    try:
        import whisper  # type: ignore
    except Exception:
        print("[erro] Biblioteca 'whisper' não encontrada. Instale com: pip install -U openai-whisper", file=sys.stderr)
        return None

    try:
        model = whisper.load_model(model_name)
        # word_timestamps ajuda a localizar melhor os limites, mesmo que não usemos cada palavra aqui
        result = model.transcribe(audio_video_path, language=language, word_timestamps=True, fp16=False)
        return result.get("segments", [])
    except Exception as e:
        print(f"[erro] Falha ao transcrever: {e}", file=sys.stderr)
        return None


def normalize_text(t: str) -> str:
    return "".join(ch.lower() for ch in t if ch.isalnum() or ch.isspace())


def build_speech_intervals(
    segments: List[Dict[str, object]],
    min_gap_merge_s: float = 0.30,
    min_segment_s: float = 0.20,
    smoothing_buffer_s: float = 0.20,
    total_duration: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Cria intervalos contínuos de fala a partir dos segmentos de transcrição.

    Une segmentos com lacunas pequenas (< min_gap_merge_s).
    """
    raw: List[Tuple[float, float]] = []
    for seg in segments:
        text = normalize_text(str(seg.get("text", ""))).strip()
        if not text:
            continue
        try:
            s = float(seg.get("start", 0.0))
            e = float(seg.get("end", 0.0))
        except Exception:
            continue
        if e <= s:
            continue
        raw.append((s, e))

    raw.sort(key=lambda x: x[0])
    merged: List[Tuple[float, float]] = []
    for s, e in raw:
        if not merged:
            merged.append((s, e))
            continue
        ps, pe = merged[-1]
        if s - pe <= min_gap_merge_s:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    # Expand each interval by smoothing buffer and merge overlaps again
    if total_duration is None:
        total_duration = merged[-1][1] if merged else 0.0
    expanded: List[Tuple[float, float]] = []
    for s, e in merged:
        s2 = max(0.0, s - smoothing_buffer_s)
        e2 = min(total_duration, e + smoothing_buffer_s)
        expanded.append((s2, e2))
    expanded.sort(key=lambda x: x[0])
    merged2: List[Tuple[float, float]] = []
    for s, e in expanded:
        if not merged2:
            merged2.append((s, e))
            continue
        ps, pe = merged2[-1]
        if s <= pe + min_gap_merge_s:
            merged2[-1] = (ps, max(pe, e))
        else:
            merged2.append((s, e))
    # Filter by minimal duration
    merged2 = [(s, e) for (s, e) in merged2 if (e - s) >= min_segment_s]
    return merged2


def concat_intervals_copy(source_path: str, intervals: List[Tuple[float, float]], output_path: str) -> int:
    """Concatena trechos (in/out) do mesmo arquivo com -c copy usando ffconcat."""
    entries: List[Dict[str, object]] = []
    keyframes = ffprobe_keyframes(source_path)
    for s, e in intervals:
        if e <= s:
            continue
        in_adj = keyframe_at_or_before(keyframes, max(s, 0.0)) if keyframes else max(s, 0.0)
        entries.append({"file": source_path, "in": in_adj, "out": e})

    if not entries:
        print("[info] Nenhum intervalo de fala válido encontrado.")
        return 0

    list_path = write_ffconcat(entries)
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
            list_path,
            "-c",
            "copy",
            output_path,
        ]
        return subprocess.run(cmd).returncode
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass


def concat_intervals_video_only(source_path: str, intervals: List[Tuple[float, float]], output_path: str) -> int:
    entries: List[Dict[str, object]] = []
    keyframes = ffprobe_keyframes(source_path)
    for s, e in intervals:
        if e <= s:
            continue
        in_adj = keyframe_at_or_before(keyframes, max(s, 0.0)) if keyframes else max(s, 0.0)
        entries.append({"file": source_path, "in": in_adj, "out": e})

    if not entries:
        return 0

    list_path = write_ffconcat(entries)
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
            list_path,
            "-an",
            "-c",
            "copy",
            output_path,
        ]
        return subprocess.run(cmd).returncode
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass


def build_cut_plan(source_path: str, intervals: List[Tuple[float, float]], crossfade_s: float) -> List[Dict[str, float]]:
    """Compute a unified cut plan so audio and video share identical timeline.

    For each interval [s,e]:
    - Align in to keyframe (adj_in)
    - base_len = e - s
    - video_out = adj_in + base_len - crossfade (for non-last)
    - audio_out = adj_in + base_len (audio keeps the overlap which acrossfade consumes)
    """
    keyframes = ffprobe_keyframes(source_path)
    dur = ffprobe_duration(source_path) or 0.0
    plan: List[Dict[str, float]] = []
    n = len(intervals)
    for i, (s, e) in enumerate(intervals):
        if e <= s:
            continue
        base_len = e - s
        adj_in = keyframe_at_or_before(keyframes, max(s, 0.0)) if keyframes else max(s, 0.0)
        v_out = adj_in + base_len - (crossfade_s if i < n - 1 else 0.0)
        a_out = adj_in + base_len
        # Clamp
        adj_in = max(0.0, min(adj_in, dur))
        v_out = max(adj_in, min(v_out, dur))
        a_out = max(adj_in, min(a_out, dur))
        plan.append({"v_in": adj_in, "v_out": v_out, "a_in": adj_in, "a_out": a_out})
    return plan


def concat_video_from_plan(source_path: str, plan: List[Dict[str, float]], output_path: str) -> int:
    entries: List[Dict[str, object]] = []
    for seg in plan:
        entries.append({"file": source_path, "in": seg["v_in"], "out": seg["v_out"]})

    if not entries:
        return 0

    list_path = write_ffconcat(entries)
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
            list_path,
            "-an",
            "-c",
            "copy",
            output_path,
        ]
        return subprocess.run(cmd).returncode
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass


def build_audio_with_crossfades_from_plan(source_path: str, plan: List[Dict[str, float]], out_audio: str, crossfade_s: float) -> int:
    if not plan:
        return 0
    # Build filter graph: atrim each interval, then acrossfade chain
    parts = []
    for i, seg in enumerate(plan):
        s = seg["a_in"]
        e = seg["a_out"]
        parts.append(f"[0:a]atrim=start={s:.6f}:end={e:.6f},asetpts=PTS-STARTPTS[a{i}]")

    # Chain acrossfades
    if len(plan) == 1:
        chain = "[a0]anull[aout]"
    else:
        labels = [f"a{i}" for i in range(len(plan))]
        out_label = "x1"
        chain_steps = [f"[{labels[0]}][{labels[1]}]acrossfade=d={crossfade_s:.3f}:c1=tri:c2=tri[{out_label}]"]
        cur = out_label
        for i in range(2, len(labels)):
            nxt = f"x{i}"
            chain_steps.append(f"[{cur}][{labels[i]}]acrossfade=d={crossfade_s:.3f}:c1=tri:c2=tri[{nxt}]")
            cur = nxt
        chain = ";".join(chain_steps) + ";" + f"[{cur}]anull[aout]"

    filter_complex = ";".join(parts) + ";" + chain
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        source_path,
        "-filter_complex",
        filter_complex,
        "-map",
        "[aout]",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        out_audio,
    ]
    return subprocess.run(cmd).returncode


def main(argv: List[str]) -> int:
    if not has_ffmpeg():
        print("ffmpeg não encontrado.", file=sys.stderr)
        return 2

    sources_dir = get_sources_dir()
    merged = os.path.join(os.getcwd(), "outputs", "final_merged_J_text.mp4")
    if len(argv) > 1:
        merged = argv[1]
    if not os.path.exists(merged):
        print(f"[erro] Arquivo não encontrado: {merged}", file=sys.stderr)
        return 1

    print(f"[info] Transcrevendo: {merged}")
    segs = transcribe_full(merged, language="pt", model_name="base")
    if segs is None:
        return 3

    print("[info] Gerando intervalos de fala...")
    intervals = build_speech_intervals(
        segs,
        min_gap_merge_s=0.30,
        min_segment_s=0.20,
        smoothing_buffer_s=0.20,
        total_duration=ffprobe_duration(merged) or 0.0,
    )
    dur = ffprobe_duration(merged) or 0.0
    # Garantir limites dentro do arquivo
    intervals = [(max(0.0, s), min(dur, e)) for s, e in intervals if e > s]
    print(f"[info] {len(intervals)} intervalos de fala detectados.")

    # Concat vídeo sem áudio e gerar áudio com crossfades, depois mux
    out_dir = os.path.join(os.getcwd(), "outputs")
    out_path = os.path.join(out_dir, "outputs_008_speech_only_smoothed.mp4")
    tmp_video = os.path.join(out_dir, "tmp_speech_video.mp4")
    tmp_audio = os.path.join(out_dir, "tmp_speech_audio.m4a")
    for p in (out_path, tmp_video, tmp_audio):
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass

    crossfade_s = 0.20
    # Plano unificado e corte exato para vídeo e áudio
    plan = build_cut_plan(merged, intervals, crossfade_s=crossfade_s)

    print(f"[info] Concatenando vídeo (copy, sem áudio) com plano unificado...")
    code = concat_video_from_plan(merged, plan, tmp_video)
    if code != 0:
        print(f"[erro] concat vídeo retornou código {code}", file=sys.stderr)
        return code

    print(f"[info] Gerando áudio com crossfades a partir do plano...")
    code = build_audio_with_crossfades_from_plan(merged, plan, tmp_audio, crossfade_s=crossfade_s)
    if code != 0:
        print(f"[erro] build áudio retornou código {code}", file=sys.stderr)
        return code

    # Conferência opcional de durações
    vdur = ffprobe_duration(tmp_video) or 0.0
    adur = ffprobe_duration(tmp_audio) or 0.0
    # Igualar duração com corte milimétrico no vídeo se necessário (sem reencode)
    if vdur > adur + 0.02:
        print(f"[info] Ajuste fino de duração: v={vdur:.3f}s > a={adur:.3f}s. Cortando vídeo no final...")
        list_path = write_ffconcat([
            {"file": tmp_video, "in": 0.0, "out": adur}
        ])
        try:
            trimmed_video = os.path.join(out_dir, "tmp_speech_video_trimmed.mp4")
            if os.path.exists(trimmed_video):
                os.remove(trimmed_video)
            cmd = [
                "ffmpeg","-hide_banner","-loglevel","error","-y",
                "-f","concat","-safe","0","-i", list_path,
                "-an","-c","copy", trimmed_video
            ]
            rc = subprocess.run(cmd).returncode
            if rc == 0:
                try:
                    os.remove(tmp_video)
                except OSError:
                    pass
                tmp_video = trimmed_video
                vdur = adur
        finally:
            try:
                os.remove(list_path)
            except OSError:
                pass

    print(f"[info] Muxando resultado -> {out_path}")
    code = mux_video_and_audio(tmp_video, tmp_audio, out_path)
    if code != 0:
        print(f"[erro] mux retornou código {code}", file=sys.stderr)
        return code
    try:
        os.remove(tmp_video)
        os.remove(tmp_audio)
    except OSError:
        pass
    print("[done] Arquivo gerado com suavização de cortes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


