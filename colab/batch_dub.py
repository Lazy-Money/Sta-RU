"""
Batch dubbing for Sta-RU.

Procesa todos los videos de un directorio, emparejándolos con su SRT por nombre.
Es resumible: si el output ya existe, salta el video.

Uso:
    python batch_dub.py \
        --videos-dir /content/drive/MyDrive/sta-ru-videos \
        --srt-dir Subtitles-EN \
        --output-dir /content/drive/MyDrive/sta-ru-output \
        --lang en

El emparejamiento se hace por el "stem" del nombre quitando el sufijo de idioma:
    video:  "2016-07-12 - 1 - Standing wave in the inductor (early 2016).mp4"
    srt:    "2016-07-12 - 1 - Standing wave in the inductor (early 2016)-EN.txt"

Si los nombres no son idénticos, podés usar --manifest con un CSV de "video_path,srt_path".
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyrubberband as pyrb
import soundfile as sf
import srt
import torch
from TTS.api import TTS


MAX_SPEED_RATIO = 1.4
SAMPLE_RATE = 24000
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


def find_srt(srt_dir: Path, video_path: Path, lang: str) -> Path | None:
    """Busca el SRT que matchea el video por nombre."""
    stem = video_path.stem
    # Intentos: exact, con -EN, con -<LANG>
    candidates = [
        srt_dir / f"{stem}-{lang.upper()}.txt",
        srt_dir / f"{stem}-{lang.upper()}.srt",
        srt_dir / f"{stem}.{lang}.srt",
        srt_dir / f"{stem}-{lang}.txt",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fuzzy: cualquier archivo que empiece con el stem
    for f in srt_dir.iterdir():
        if f.stem.startswith(stem) and f.suffix in {".txt", ".srt"}:
            return f
    return None


def extract_voice_ref(orig_audio_path: Path, subs: list[srt.Subtitle], out_path: Path) -> tuple[float, float]:
    """Extrae sample de voz: el segmento más largo entre 6-15s."""
    orig, sr = sf.read(orig_audio_path)
    candidates = [
        s for s in subs
        if 6 <= (s.end - s.start).total_seconds() <= 15
    ]
    if candidates:
        ref = max(candidates, key=lambda s: (s.end - s.start).total_seconds())
        start_s = ref.start.total_seconds()
        end_s = ref.end.total_seconds()
    else:
        start_s, end_s = 1.0, 11.0

    ref_audio = orig[int(start_s * sr):int(end_s * sr)]
    sf.write(out_path, ref_audio, sr)
    return start_s, end_s


def get_video_duration(video_path: Path) -> float:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(probe.stdout.strip())


def extract_audio(video_path: Path, out_path: Path):
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video_path), "-ac", "1", "-ar", str(SAMPLE_RATE),
            str(out_path),
        ],
        check=True,
    )


def time_fit(audio: np.ndarray, sr: int, target_duration: float) -> np.ndarray:
    actual = len(audio) / sr
    if actual <= target_duration * 1.02:
        return audio
    speed = actual / target_duration
    speed = min(speed, MAX_SPEED_RATIO)
    return pyrb.time_stretch(audio, sr, speed)


def dub_one(
    video_path: Path,
    srt_path: Path,
    output_path: Path,
    tts: TTS,
    lang: str,
    work_dir: Path,
):
    print(f"\n=== {video_path.name} ===")
    work_dir.mkdir(parents=True, exist_ok=True)
    seg_dir = work_dir / "segments"
    seg_dir.mkdir(exist_ok=True)

    # 1. Parse SRT
    subs = list(srt.parse(srt_path.read_text(encoding="utf-8")))
    print(f"  {len(subs)} segmentos")

    # 2. Extract original audio + voice reference
    orig_audio = work_dir / "orig.wav"
    extract_audio(video_path, orig_audio)
    ref_path = work_dir / "ref.wav"
    ref_start, ref_end = extract_voice_ref(orig_audio, subs, ref_path)
    print(f"  Voz ref: {ref_start:.1f}s-{ref_end:.1f}s ({ref_end-ref_start:.1f}s)")

    # 3. Generate per-segment
    print("  Generando audio...", flush=True)
    fitted = []
    failed = 0
    for i, sub in enumerate(subs):
        text = sub.content.strip()
        if not text:
            fitted.append(None)
            continue
        seg_path = seg_dir / f"seg_{i:04d}.wav"
        try:
            tts.tts_to_file(
                text=text,
                file_path=str(seg_path),
                speaker_wav=str(ref_path),
                language=lang,
                split_sentences=False,
            )
            audio, sr = sf.read(seg_path)
            target = (sub.end - sub.start).total_seconds()
            fitted.append((time_fit(audio, sr, target), sr))
        except Exception as e:
            print(f"  [WARN] seg {i+1}: {e}")
            fitted.append(None)
            failed += 1
    print(f"  Generados: {sum(1 for f in fitted if f)}/{len(subs)} (fallos: {failed})")

    # 4. Build master audio
    video_duration = get_video_duration(video_path)
    sr_master = next(f[1] for f in fitted if f is not None)
    master = np.zeros(int(video_duration * sr_master) + sr_master, dtype=np.float32)
    for sub, fit in zip(subs, fitted):
        if fit is None:
            continue
        audio, _ = fit
        start_sample = int(sub.start.total_seconds() * sr_master)
        end_sample = min(start_sample + len(audio), len(master))
        master[start_sample:end_sample] += audio[:end_sample - start_sample].astype(np.float32)
    peak = float(np.max(np.abs(master)))
    if peak > 0.99:
        master *= 0.99 / peak

    master_path = work_dir / "master.wav"
    sf.write(master_path, master, sr_master)

    # 5. Mux
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video_path),
            "-i", str(master_path),
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest",
            str(output_path),
        ],
        check=True,
    )
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  ✓ {output_path.name} ({size_mb:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", type=Path, required=True)
    ap.add_argument("--srt-dir", type=Path, default=Path("Subtitles-EN"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--work-dir", type=Path, default=Path("/tmp/sta-ru-work"))
    ap.add_argument("--manifest", type=Path, help="CSV opcional con video_path,srt_path por fila")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    # Build (video, srt) list
    pairs = []
    if args.manifest:
        with args.manifest.open() as f:
            for row in csv.DictReader(f):
                pairs.append((Path(row["video_path"]), Path(row["srt_path"])))
    else:
        videos = sorted(
            p for p in args.videos_dir.iterdir()
            if p.suffix.lower() in VIDEO_EXTS
        )
        for v in videos:
            s = find_srt(args.srt_dir, v, args.lang)
            if s is None:
                print(f"[SKIP] sin SRT: {v.name}")
                continue
            pairs.append((v, s))

    if not pairs:
        print("No hay pares (video, srt) para procesar.")
        sys.exit(1)
    print(f"A procesar: {len(pairs)} videos\n")

    # Filter already-done
    to_do = []
    for v, s in pairs:
        out = args.output_dir / f"{v.stem}-{args.lang.upper()}.mp4"
        if args.skip_existing and out.exists():
            print(f"[OK ya existe] {out.name}")
            continue
        to_do.append((v, s, out))
    print(f"Pendientes: {len(to_do)}\n")
    if not to_do:
        return

    # Load TTS once
    os.environ["COQUI_TOS_AGREED"] = "1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cargando XTTS-v2 en {device}...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print("Listo.\n")

    for i, (video, srt_path, out) in enumerate(to_do):
        print(f"\n[{i+1}/{len(to_do)}] {video.name}")
        try:
            dub_one(
                video_path=video,
                srt_path=srt_path,
                output_path=out,
                tts=tts,
                lang=args.lang,
                work_dir=args.work_dir / video.stem,
            )
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            continue

    print(f"\n=== Done. Outputs en {args.output_dir} ===")


if __name__ == "__main__":
    main()
