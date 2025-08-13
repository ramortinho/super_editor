import os
import sys
import time
from typing import List, Optional

from mesclar import (
    list_video_files_by_mtime,
    ffprobe_streams,
    ensure_same_params,
    safe_remove,
    get_sources_dir,
    merge_two_j,  # fallback (pico de áudio)
)
from mesclar_text import merge_two_j_text  # preferido (transcrição)


def format_seconds(total_seconds: float) -> str:
    total_seconds = max(0, int(total_seconds))
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def merge_all_sources_j_text(sources_dir: str, output_path: Optional[str] = None) -> int:
    files: List[str] = list_video_files_by_mtime(sources_dir)
    if len(files) < 2:
        print("Nenhum trabalho a fazer: é necessário pelo menos 2 vídeos em 'sources/'.", file=sys.stderr)
        return 1

    # Validar parâmetros compatíveis (codecs, fps, resolução)
    print(f"[info] Verificando compatibilidade de {len(files)} arquivos...")
    metas = []
    for p in files:
        m = ffprobe_streams(p)
        if m is None:
            print(f"[erro] Falha ao inspecionar metadados: {p}", file=sys.stderr)
            return 2
        metas.append(m)
    for i in range(len(metas) - 1):
        ok, reason = ensure_same_params(metas[i], metas[i + 1])
        if not ok:
            print(f"[erro] Incompatível entre {os.path.basename(files[i])} e {os.path.basename(files[i+1])}: {reason}", file=sys.stderr)
            return 3

    outputs_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(outputs_dir, "final_merged_J_text.mp4")

    tmp_a = os.path.join(outputs_dir, "tmp_final_a.mp4")
    tmp_b = os.path.join(outputs_dir, "tmp_final_b.mp4")
    safe_remove(tmp_a)
    safe_remove(tmp_b)
    safe_remove(output_path)

    total_merges = len(files) - 1
    started_at = time.time()
    merges_done = 0

    # Inicializa com o primeiro arquivo
    current_path = files[0]

    print(f"[info] Iniciando mesclagem de {len(files)} arquivos (J/Texto). Saída: {output_path}")

    for idx in range(1, len(files)):
        nxt = files[idx]
        target_tmp = tmp_a if (idx % 2 == 1) else tmp_b
        safe_remove(target_tmp)

        print(f"[step {idx}/{total_merges}] {os.path.basename(current_path)} + {os.path.basename(nxt)}")
        step_start = time.time()

        # Preferir mesclagem por texto (última palavra); fallback para pico
        code = merge_two_j_text(
            current_path,
            nxt,
            phrase=None,
            overlap=1.0,
            window_seconds=3.0,
            exclude_tail_seconds=1.0,
            output_path=target_tmp,
        )
        if code != 0:
            print("[warn] Falha na mesclagem por texto. Tentando fallback por pico...")
            code = merge_two_j(current_path, nxt, overlap=1.0, out_path=target_tmp)
            if code != 0:
                print("[erro] Falha na mesclagem por pico também.", file=sys.stderr)
                return code

        current_path = target_tmp

        merges_done += 1
        elapsed = time.time() - started_at
        avg = elapsed / max(1, merges_done)
        remaining_merges = total_merges - merges_done
        eta_seconds = remaining_merges * avg
        step_elapsed = time.time() - step_start
        print(
            f"[ok] Merge {merges_done}/{total_merges} concluído em {format_seconds(int(step_elapsed))}. "
            f"Tempo total: {format_seconds(int(elapsed))}. Restante (estimado): {format_seconds(int(eta_seconds))}."
        )

    # current_path contém o último tmp. Renomear para destino final
    if os.path.abspath(current_path) != os.path.abspath(output_path):
        try:
            safe_remove(output_path)
            os.replace(current_path, output_path)
        except Exception:
            # fallback copy
            import shutil as _sh
            _sh.copyfile(current_path, output_path)

    # limpeza de temporários sobressalentes
    safe_remove(tmp_a)
    safe_remove(tmp_b)

    print(f"[done] Mesclagem final concluída: {output_path}")
    return 0


def main(argv: List[str]) -> int:
    sources_dir = get_sources_dir()
    return merge_all_sources_j_text(sources_dir)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


