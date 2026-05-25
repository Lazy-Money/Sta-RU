"""
Sta-RU Video Dubbing — batch script.

Reads YouTube URLs (pasted, .txt or .csv), pairs each with a user-uploaded
SRT named {N#}-{LANG}.srt, and produces dubbed videos with cloned voice.

Designed to be called from the Colab notebook (`Sta_RU_Dubbing.ipynb`)
via `run_batch(...)`, but also works as a standalone CLI.

Pipeline per video:
    1. Fetch metadata + download video (yt-dlp)
    2. Extract original audio
    3. Sample voice reference from longest segment (6-15s)
    4. Optional: strip vocals from original audio (Demucs)
    5. Generate TTS per SRT segment (XTTS-v2, voice-cloned)
    6. Time-fit each segment (rubberband, max 1.4x)
    7. Mix TTS into ambient (or silence) on the original video timeline
    8. Mux with video → {Final Name}-{LANG}.{ext}
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pyrubberband as pyrb
import soundfile as sf
import srt
import torch
from TTS.api import TTS


# ============================================================
#  Constants
# ============================================================
MAX_SPEED_RATIO = 1.4
SAMPLE_RATE = 24000

# XTTS-v2 supported languages (ISO 639-1, except zh)
XTTS_LANGS = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
    "nl", "cs", "ar", "zh", "hu", "ko", "ja", "hi",
}

# Windows-forbidden chars in filenames
_FORBIDDEN = re.compile(r'[\\/:*?"<>|]')


# ============================================================
#  Data classes
# ============================================================
@dataclass
class VideoItem:
    n: int                       # ordinal (1-based)
    url: str
    upload_date: str = ""        # YYYY-MM-DD
    title: str = ""              # original title from YouTube
    duration: float = 0.0        # seconds
    title_translated: str = ""   # optional translation to target lang
    srt_path: Path | None = None
    output_path: Path | None = None
    status: str = "pending"      # pending | done | skipped | failed
    error: str = ""


# ============================================================
#  URL loading
# ============================================================
def load_urls(source: str | Path | list[str]) -> list[str]:
    """Load URLs from a list, .txt or .csv file, or a multiline string."""
    if isinstance(source, list):
        urls = source
    elif isinstance(source, (str, Path)) and Path(source).is_file():
        path = Path(source)
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                first = next(reader, None)
                # Treat as headerless if first row's first cell looks like a URL
                if first and first[0].strip().startswith("http"):
                    urls = [first[0].strip()]
                else:
                    urls = []
                for row in reader:
                    if row and row[0].strip():
                        urls.append(row[0].strip())
        else:
            urls = path.read_text(encoding="utf-8").splitlines()
    else:
        urls = str(source).splitlines()

    # Clean: strip, drop empties and comments
    clean = []
    for u in urls:
        u = u.strip()
        if not u or u.startswith("#"):
            continue
        if not u.startswith("http"):
            print(f"[WARN] skipping non-URL line: {u[:60]}")
            continue
        clean.append(u)
    return clean


# ============================================================
#  Metadata via yt-dlp
# ============================================================
def fetch_metadata(url: str) -> dict:
    """Fetch video metadata without downloading. Returns {} on failure."""
    try:
        r = subprocess.run(
            ["yt-dlp", "--no-warnings", "--skip-download", "--dump-json", url],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            return {"_error": r.stderr.strip().splitlines()[-1] if r.stderr else "unknown"}
        return json.loads(r.stdout)
    except subprocess.TimeoutExpired:
        return {"_error": "timeout"}
    except Exception as e:
        return {"_error": str(e)}


def _fmt_date(raw: str) -> str:
    """yt-dlp gives 'YYYYMMDD'; we want 'YYYY-MM-DD'."""
    if raw and len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    return raw or ""


def build_items(urls: list[str], translate_titles: bool, target_lang: str) -> list[VideoItem]:
    """Build VideoItem list with metadata for each URL."""
    items: list[VideoItem] = []
    for i, url in enumerate(urls, start=1):
        meta = fetch_metadata(url)
        item = VideoItem(n=i, url=url)
        if "_error" in meta:
            item.status = "failed"
            item.error = f"metadata: {meta['_error']}"
            items.append(item)
            continue
        item.title = meta.get("title", "") or ""
        item.upload_date = _fmt_date(meta.get("upload_date", "") or "")
        item.duration = float(meta.get("duration") or 0.0)
        items.append(item)

    if translate_titles and any(it.title for it in items):
        _translate_titles_inplace(items, target_lang)
    return items


def _translate_titles_inplace(items: list[VideoItem], target_lang: str) -> None:
    """Translate each item.title → item.title_translated using deep-translator."""
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        print("[WARN] deep-translator not installed; titles will not be translated")
        return
    translator = GoogleTranslator(source="auto", target=target_lang)
    for it in items:
        if not it.title:
            continue
        try:
            it.title_translated = (translator.translate(it.title) or "").strip()
        except Exception as e:
            print(f"[WARN] N#{it.n}: title translation failed ({e})")


# ============================================================
#  Filename handling
# ============================================================
def sanitize(name: str) -> str:
    """Make a string safe for use as a filename on Windows/macOS/Linux."""
    out = _FORBIDDEN.sub("_", name).strip().rstrip(". ")
    return out or "untitled"


def build_output_name(item: VideoItem, lang: str, translate: bool, ext: str = "mp4") -> str:
    """{date} - {N#} - {title (translated or original)}-{LANG}.{ext}"""
    base_title = item.title_translated if (translate and item.title_translated) else item.title
    base_title = sanitize(base_title or f"video_{item.n}")
    date_prefix = f"{item.upload_date} - " if item.upload_date else ""
    return f"{date_prefix}{item.n} - {base_title}-{lang.upper()}.{ext}"


# ============================================================
#  Range filtering
# ============================================================
def parse_range(expr: str, max_n: int) -> set[int]:
    """Parse 'all', '1-10', '47', '1-5,12,20-25' → set of N#."""
    expr = (expr or "").strip().lower()
    if not expr or expr == "all":
        return set(range(1, max_n + 1))
    selected: set[int] = set()
    for chunk in expr.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            selected.update(range(int(a), int(b) + 1))
        else:
            selected.add(int(chunk))
    return selected & set(range(1, max_n + 1))


# ============================================================
#  ffmpeg / yt-dlp helpers
# ============================================================
def download_video(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "yt-dlp",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "--no-warnings",
            "-o", str(out_path), url,
        ],
        check=True,
    )


def extract_audio(video_path: Path, out_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video_path), "-ac", "1", "-ar", str(SAMPLE_RATE),
            str(out_path),
        ],
        check=True,
    )


def get_video_duration(video_path: Path) -> float:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(probe.stdout.strip())


# ============================================================
#  Voice reference extraction
# ============================================================
def extract_voice_ref(audio_path: Path, subs: list[srt.Subtitle], out_path: Path) -> tuple[float, float]:
    audio, sr = sf.read(audio_path)
    candidates = [s for s in subs if 6 <= (s.end - s.start).total_seconds() <= 15]
    if candidates:
        ref = max(candidates, key=lambda s: (s.end - s.start).total_seconds())
        start_s, end_s = ref.start.total_seconds(), ref.end.total_seconds()
    else:
        start_s, end_s = 1.0, 11.0
    sf.write(out_path, audio[int(start_s * sr):int(end_s * sr)], sr)
    return start_s, end_s


# ============================================================
#  Demucs vocal removal (optional)
# ============================================================
def strip_vocals(audio_path: Path, out_path: Path) -> bool:
    """Run Demucs to remove vocals; writes the 'no_vocals' stem to out_path.
    Returns True on success, False if Demucs not installed or fails."""
    try:
        import demucs.separate  # noqa: F401
    except ImportError:
        print("[WARN] demucs not installed; skipping vocal removal")
        return False
    work = audio_path.parent / "demucs_out"
    work.mkdir(exist_ok=True)
    try:
        subprocess.run(
            [
                sys.executable, "-m", "demucs.separate",
                "--two-stems", "vocals",
                "-n", "htdemucs",
                "-o", str(work),
                str(audio_path),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[WARN] demucs failed: {e.stderr.decode()[-200:] if e.stderr else e}")
        return False
    # demucs writes to {work}/htdemucs/{stem}/no_vocals.wav
    no_vocals = next(work.rglob("no_vocals.wav"), None)
    if not no_vocals:
        return False
    shutil.copy(no_vocals, out_path)
    shutil.rmtree(work, ignore_errors=True)
    return True


# ============================================================
#  TTS + time-fit
# ============================================================
def time_fit(audio: np.ndarray, sr: int, target_duration: float) -> tuple[np.ndarray, bool]:
    """Compress audio to fit target_duration (no expansion). Returns (audio, capped)."""
    actual = len(audio) / sr
    if actual <= target_duration * 1.02:
        return audio, False
    speed = actual / target_duration
    capped = speed > MAX_SPEED_RATIO
    speed = min(speed, MAX_SPEED_RATIO)
    return pyrb.time_stretch(audio, sr, speed), capped


def dub_one(
    item: VideoItem,
    tts: TTS,
    lang: str,
    work_dir: Path,
    output_path: Path,
    remove_voice: bool,
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    seg_dir = work_dir / "segments"
    seg_dir.mkdir(exist_ok=True)

    # Parse SRT
    subs = list(srt.parse(item.srt_path.read_text(encoding="utf-8")))
    print(f"  SRT: {len(subs)} segments")

    # Download video
    video_path = work_dir / "video.mp4"
    if not video_path.exists():
        print("  Downloading video...", flush=True)
        download_video(item.url, video_path)

    # Extract audio
    orig_audio = work_dir / "orig.wav"
    extract_audio(video_path, orig_audio)

    # Voice reference (always from voiced audio, pre-demucs)
    ref_path = work_dir / "ref.wav"
    ref_s, ref_e = extract_voice_ref(orig_audio, subs, ref_path)
    print(f"  Voice ref: {ref_s:.1f}-{ref_e:.1f}s")

    # Optional: vocal removal for ambient bed
    ambient_path: Path | None = None
    if remove_voice:
        print("  Stripping vocals (Demucs)...", flush=True)
        stripped = work_dir / "ambient.wav"
        if strip_vocals(orig_audio, stripped):
            ambient_path = stripped

    # Generate TTS per segment
    print("  Generating TTS...", flush=True)
    fitted: list[tuple[np.ndarray, int] | None] = []
    n_capped = 0
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
            audio, sr_seg = sf.read(seg_path)
            target = (sub.end - sub.start).total_seconds()
            audio, capped = time_fit(audio, sr_seg, target)
            n_capped += int(capped)
            fitted.append((audio, sr_seg))
        except Exception as e:
            print(f"  [WARN] seg {i+1}: {e}")
            fitted.append(None)
    generated = sum(1 for f in fitted if f is not None)
    print(f"  Generated: {generated}/{len(subs)} (capped at {MAX_SPEED_RATIO}x: {n_capped})")

    # Build master track
    video_duration = get_video_duration(video_path)
    sr_master = next(f[1] for f in fitted if f is not None)
    total_samples = int(video_duration * sr_master) + sr_master
    master = np.zeros(total_samples, dtype=np.float32)
    for sub, fit in zip(subs, fitted):
        if fit is None:
            continue
        audio, _ = fit
        start = int(sub.start.total_seconds() * sr_master)
        end = min(start + len(audio), len(master))
        master[start:end] += audio[:end - start].astype(np.float32)

    # Mix ambient if available
    if ambient_path is not None:
        amb, sr_amb = sf.read(ambient_path)
        if amb.ndim > 1:
            amb = amb.mean(axis=1)
        if sr_amb != sr_master:
            from scipy import signal
            amb = signal.resample_poly(amb, sr_master, sr_amb)
        amb = amb[:len(master)] if len(amb) >= len(master) else np.pad(amb, (0, len(master) - len(amb)))
        # Ambient at -12dB so voice sits on top
        master = master + amb.astype(np.float32) * 0.25

    # Normalize
    peak = float(np.max(np.abs(master)))
    if peak > 0.99:
        master *= 0.99 / peak

    master_path = work_dir / "master.wav"
    sf.write(master_path, master, sr_master)

    # Mux
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

    # Cleanup work_dir to save disk
    shutil.rmtree(work_dir, ignore_errors=True)


# ============================================================
#  Public entry point
# ============================================================
def run_batch(
    urls: str | Path | list[str],
    srt_dir: str | Path,
    output_dir: str | Path,
    lang: str = "EN",
    translate_titles: bool = False,
    remove_voice: bool = True,
    range_expr: str = "all",
    work_root: str | Path = "/tmp/sta-ru-work",
) -> list[VideoItem]:
    """Main entry point. Returns the items list with final statuses."""
    lang_lc = lang.lower()
    if lang_lc not in XTTS_LANGS:
        raise ValueError(f"Language '{lang}' not supported. Supported: {sorted(XTTS_LANGS)}")

    url_list = load_urls(urls)
    if not url_list:
        print("No URLs to process.")
        return []
    print(f"\n{'='*60}\nLoaded {len(url_list)} URLs\n{'='*60}\n")

    print("Fetching metadata from YouTube...")
    items = build_items(url_list, translate_titles=translate_titles, target_lang=lang_lc)

    # Pair with SRTs and resolve output paths
    srt_dir_path = Path(srt_dir)
    output_dir_path = Path(output_dir)
    for it in items:
        if it.status == "failed":
            continue
        srt_path = srt_dir_path / f"{it.n}-{lang.upper()}.srt"
        if not srt_path.exists():
            it.status = "skipped"
            it.error = f"no SRT: {srt_path.name}"
            continue
        it.srt_path = srt_path
        out_name = build_output_name(it, lang, translate_titles, ext="mp4")
        it.output_path = output_dir_path / out_name

    # Range filter
    selected = parse_range(range_expr, len(items))
    for it in items:
        if it.n not in selected and it.status == "pending":
            it.status = "skipped"
            it.error = "out of range"

    # Skip already done
    for it in items:
        if it.status == "pending" and it.output_path and it.output_path.exists():
            it.status = "done"
            it.error = "already exists"

    # Pretty print plan
    print(f"\n{'='*60}\nPlan\n{'='*60}")
    print(f"{'N#':>3} {'Status':>8}  Title")
    for it in items:
        title = it.title_translated if (translate_titles and it.title_translated) else it.title
        title = title[:60] or "(no metadata)"
        print(f"{it.n:>3} {it.status:>8}  {title}")
    pending = [it for it in items if it.status == "pending"]
    print(f"\nPending: {len(pending)} | Done already: {sum(1 for i in items if i.status == 'done')} | Skipped: {sum(1 for i in items if i.status == 'skipped')} | Failed: {sum(1 for i in items if i.status == 'failed')}")

    if not pending:
        return items

    # Load TTS once
    os.environ["COQUI_TOS_AGREED"] = "1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading XTTS-v2 on {device}...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print("Model ready.\n")

    work_root_path = Path(work_root)
    for idx, it in enumerate(pending, start=1):
        print(f"\n[{idx}/{len(pending)}] N#{it.n} — {(it.title or '')[:60]}")
        t0 = time.time()
        try:
            dub_one(
                item=it,
                tts=tts,
                lang=lang_lc,
                work_dir=work_root_path / f"n{it.n}",
                output_path=it.output_path,
                remove_voice=remove_voice,
            )
            it.status = "done"
            print(f"  Elapsed: {time.time() - t0:.1f}s")
        except Exception as e:
            it.status = "failed"
            it.error = str(e)
            print(f"  ✗ FAILED: {e}")

    # Final summary
    print(f"\n{'='*60}\nSummary\n{'='*60}")
    for s in ("done", "skipped", "failed"):
        ns = [it.n for it in items if it.status == s]
        if ns:
            print(f"  {s:>8} ({len(ns)}): {ns}")
    print(f"\nOutputs in: {output_dir_path}")
    return items


# ============================================================
#  CLI
# ============================================================
def _cli() -> None:
    ap = argparse.ArgumentParser(description="Sta-RU batch dubbing")
    ap.add_argument("--urls", required=True, help="Path to .txt/.csv with URLs (one per line/row)")
    ap.add_argument("--srt-dir", required=True, help="Folder containing {N#}-{LANG}.srt files")
    ap.add_argument("--output-dir", required=True, help="Where to write dubbed videos")
    ap.add_argument("--lang", default="EN", help="Target language code (EN, ES, DE, IT, ...)")
    ap.add_argument("--translate-titles", action="store_true", help="Translate titles to target language")
    ap.add_argument("--no-remove-voice", action="store_true", help="Skip Demucs vocal removal")
    ap.add_argument("--range", dest="range_expr", default="all", help="e.g. 'all', '1-10', '47', '1-5,12'")
    ap.add_argument("--work-root", default="/tmp/sta-ru-work")
    args = ap.parse_args()

    run_batch(
        urls=args.urls,
        srt_dir=args.srt_dir,
        output_dir=args.output_dir,
        lang=args.lang,
        translate_titles=args.translate_titles,
        remove_voice=not args.no_remove_voice,
        range_expr=args.range_expr,
        work_root=args.work_root,
    )


if __name__ == "__main__":
    _cli()
