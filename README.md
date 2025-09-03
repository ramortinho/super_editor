# super_editor

Editor de vídeo em Python com:
- Listagem de mídia por data/formato/FPS/resolução
- Mesclas com curvas J/L (stream copy de vídeo + crossfade de áudio)
- Mescla guiada por texto (Whisper) detectando a última palavra para corte
- Pipeline final de mesclagem de toda a pasta `sources/` com logs e ETA
- Corte “somente falas” com plano unificado A/V (sincronismo matemático)
- Criação de intro: batidas de música + trechos de fala “energéticos” com música a −5 dB

## Pré‑requisitos
- FFmpeg (ffmpeg/ffprobe) instalado e no PATH
- Python 3.13+
- Windows PowerShell (ou shell equivalente)

## Instalação (primeira vez)
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

Observação: se você tiver GPU e quiser acelerar a transcrição Whisper, instale o PyTorch compatível com sua placa no lugar do pacote CPU.

## Estrutura dos principais scripts
- `listar.py`: lista os vídeos em `sources/` ordenados por data, exibindo formato, FPS e resolução
- `mesclar.py`: concatenação de dois clipes com curvas J/L (vídeo em copy, áudio com fades), alinhando cortes a keyframes
- `mesclar_text.py`: mescla J usando transcrição (Whisper) para localizar a última palavra dentro da janela final do clipe e cortar em cima dela
- `mesclar_final.py`: mescla sequencialmente todos os vídeos da pasta `sources/`, com logs e estimativa de tempo restante
- `cortar.py`: gera uma versão “somente falas” a partir do vídeo final; aplica um plano unificado de cortes para áudio e vídeo (mesmos in/out), contabilizando matematicamente o crossfade – elimina descompassos
- `intro.py`: analisa batidas de uma música em `sounds/` e posiciona trechos de fala energéticos do vídeo final nessas batidas; reduz volume da música em −5 dB e gera um mix de intro

Saídas são gravadas em `outputs/`.

## Uso rápido
1) Listar os vídeos da pasta `sources/`:
```powershell
python listar.py
```

2) Mesclar dois primeiros (teste rápido, stream copy):
```powershell
python mesclar.py
```

3) Mesclar por texto (exemplo GX015500 + GX015501, já configurado no script):
```powershell
python mesclar_text.py
```

4) Mescla completa da pasta `sources/` (gera `outputs/final_merged_J_text.mp4`):
```powershell
python mesclar_final.py
```

5) Cortes “somente fala” a partir do final mesclado, com suavização e A/V lock (gera `outputs/outputs_008_speech_only_smoothed.mp4`):
```powershell
python cortar.py
```

6) Intro: usa `sounds/03.04 - Introdução e Shorts (Remix).mp3`, detecta batidas, escolhe falas energéticas e gera `outputs/intro_audio_mix.m4a`:
```powershell
python intro.py
```

Você pode passar caminhos manualmente:
```powershell
python intro.py "sounds/MinhaMusica.mp3" outputs/final_merged_J_text.mp4
```

## Notas de sincronismo e FPS
- Vídeos GoPro frequentemente são VFR; com stream copy, o FPS exibido por alguns NLEs pode aparecer diferente (p.ex. 30.45). É esperado e o YouTube reencoda normalmente.
- Para forçar CFR 29.97 antes de subir, reexporte: `-vsync cfr -r 30000/1001`.
- `cortar.py` aplica um plano unificado: os mesmos cortes (in/out) são usados para áudio e vídeo; o crossfade é contabilizado na duração do vídeo – evitando “vídeo sobrando sem áudio”.

## Dependências
- `openai-whisper` + `torch` (ASR)
- `opencv-python` (opcional; fallback no `listar.py` quando `ffprobe` não está disponível)

## Git ignore
Já ignoramos:
- `.venv/`, `__pycache__/`, `sources/`, `outputs/`, `sounds/`, artefatos de SO e caches Python
