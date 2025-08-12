import os
import sys
import json
import shutil
from datetime import datetime
from fractions import Fraction
from typing import Dict, Optional, List, Tuple


def get_sources_dir() -> str:
    """Return the absolute path to the `sources` directory.

    Defaults to a `sources` folder in the current working directory.
    """
    default_dir = os.path.join(os.getcwd(), "sources")
    return os.environ.get("SOURCES_DIR", default_dir)


def list_video_files_by_mtime(directory: str) -> List[str]:
    """List video files in a directory, sorted by last modified time ascending.

    Only considers common video extensions. Adjust as needed.
    """
    if not os.path.isdir(directory):
        return []

    video_exts = {
        ".mp4",
        ".mov",
        ".mkv",
        ".avi",
        ".m4v",
        ".webm",
        ".wmv",
        ".flv",
    }

    def is_video(path: str) -> bool:
        return os.path.splitext(path)[1].lower() in video_exts

    entries: List[Tuple[float, str]] = []
    for name in os.listdir(directory):
        full = os.path.join(directory, name)
        if not os.path.isfile(full):
            continue
        if not is_video(full):
            continue
        try:
            mtime = os.path.getmtime(full)
        except OSError:
            # Skip unreadable files
            continue
        entries.append((mtime, full))

    entries.sort(key=lambda x: x[0])
    return [path for _, path in entries]


def has_ffprobe() -> bool:
    return shutil.which("ffprobe") is not None


def probe_with_ffprobe(path: str) -> Optional[Dict[str, object]]:
    """Extract video metadata using ffprobe.

    Returns a dict with keys: container_format, width, height, fps.
    """
    import subprocess

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
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        video_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        width = int(video_stream.get("width")) if video_stream and video_stream.get("width") else None
        height = int(video_stream.get("height")) if video_stream and video_stream.get("height") else None

        fps_value: Optional[float] = None
        if video_stream is not None:
            # Prefer avg_frame_rate; fallback to r_frame_rate
            fr = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")
            if isinstance(fr, str) and fr != "0/0":
                try:
                    fps_value = float(Fraction(fr))
                except Exception:
                    pass

        fmt = data.get("format", {})
        container_format = fmt.get("format_name") or os.path.splitext(path)[1].lstrip(".")

        return {
            "container_format": container_format,
            "width": width,
            "height": height,
            "fps": fps_value,
        }
    except Exception:
        return None


def probe_with_opencv(path: str) -> Optional[Dict[str, object]]:
    """Extract video metadata using OpenCV if available.

    Returns a dict with keys: container_format, width, height, fps.
    """
    try:
        import cv2  # type: ignore
    except Exception:
        return None

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_value = float(cap.get(cv2.CAP_PROP_FPS)) or None
        container_format = os.path.splitext(path)[1].lstrip(".")
        return {
            "container_format": container_format,
            "width": width,
            "height": height,
            "fps": fps_value,
        }
    finally:
        cap.release()


def human_dt_from_mtime(mtime: float) -> str:
    dt = datetime.fromtimestamp(mtime)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def discover_metadata(path: str) -> Dict[str, object]:
    mtime = os.path.getmtime(path)
    meta: Optional[Dict[str, object]] = None

    if has_ffprobe():
        meta = probe_with_ffprobe(path)
    if meta is None:
        meta = probe_with_opencv(path)

    if meta is None:
        # Last resort: fill minimal info
        meta = {
            "container_format": os.path.splitext(path)[1].lstrip("."),
            "width": None,
            "height": None,
            "fps": None,
        }

    out: Dict[str, object] = {
        "path": path,
        "filename": os.path.basename(path),
        "modified_time": mtime,
        "modified_time_human": human_dt_from_mtime(mtime),
        "container_format": meta.get("container_format"),
        "width": meta.get("width"),
        "height": meta.get("height"),
        "fps": meta.get("fps"),
    }
    return out


def print_table(rows: List[Dict[str, object]]) -> None:
    if not rows:
        print("Nenhum arquivo de vídeo encontrado em 'sources/'.")
        return

    # Compute column widths
    headers = ["Data", "Arquivo", "Formato", "FPS", "Resolução"]
    data_col = [r["modified_time_human"] for r in rows]
    file_col = [r["filename"] for r in rows]
    fmt_col = [str(r.get("container_format") or "?") for r in rows]
    fps_col = [
        ("{:.3f}".format(r["fps"]) if isinstance(r.get("fps"), (int, float)) and r.get("fps") else "?")
        for r in rows
    ]
    res_col = [
        (f"{r['width']}x{r['height']}" if r.get("width") and r.get("height") else "?")
        for r in rows
    ]

    widths = [
        max(len(headers[0]), max(len(str(v)) for v in data_col)),
        max(len(headers[1]), max(len(str(v)) for v in file_col)),
        max(len(headers[2]), max(len(str(v)) for v in fmt_col)),
        max(len(headers[3]), max(len(str(v)) for v in fps_col)),
        max(len(headers[4]), max(len(str(v)) for v in res_col)),
    ]

    def fmt_row(values: List[str]) -> str:
        return (
            values[0].ljust(widths[0])
            + "  "
            + values[1].ljust(widths[1])
            + "  "
            + values[2].ljust(widths[2])
            + "  "
            + values[3].rjust(widths[3])
            + "  "
            + values[4].ljust(widths[4])
        )

    print(fmt_row(headers))
    print("-" * (sum(widths) + 8))
    for i, r in enumerate(rows, start=1):
        print(
            fmt_row(
                [
                    str(r["modified_time_human"]),
                    str(r["filename"]),
                    str(r.get("container_format") or "?"),
                    ("{:.3f}".format(r["fps"]) if isinstance(r.get("fps"), (int, float)) and r.get("fps") else "?"),
                    (f"{r['width']}x{r['height']}" if r.get("width") and r.get("height") else "?"),
                ]
            )
        )


def main(argv: List[str]) -> int:
    # Directory can be overridden via CLI arg 1
    directory = argv[1] if len(argv) > 1 else get_sources_dir()

    files = list_video_files_by_mtime(directory)
    rows = [discover_metadata(p) for p in files]
    print_table(rows)

    if files and not has_ffprobe():
        # Helpful guidance if ffprobe is absent (and OpenCV might also be absent)
        print(
            "\nDica: instale o FFmpeg para metadados completos (inclui ffprobe) ou instale OpenCV:\n"
            "- FFmpeg: https://ffmpeg.org/ (garanta que 'ffprobe' esteja no PATH)\n"
            "- OpenCV (alternativa): pip install opencv-python\n",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


