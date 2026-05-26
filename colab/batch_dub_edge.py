"""
Sta-RU Video Dubbing — Edge-TTS variant.

Uses Microsoft Edge's neural TTS engine (free, no API key) via the `edge-tts`
package. Trades XTTS-v2's voice cloning for higher TTS speed, stable timing,
and better audio quality with stock neural voices.

Shares the surrounding pipeline with batch_dub.py (yt-dlp download, Demucs
ambient extraction, SRT parsing, ffmpeg mux). The only thing that changes
is how each SRT segment is synthesized.

Per-segment fit strategy:
    1. Render the line at rate=+0% (natural)
    2. If it lasts longer than the SRT slot, re-render with a rate boost
       (up to +50%) so it fits — uses Microsoft's native rate control,
       which sounds far cleaner than time-stretching a generated waveform.
    3. If it still doesn't fit, accept a small overflow (no clipping).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import srt

# Jupyter/Colab already runs an event loop, so plain asyncio.run() raises
# "cannot be called from a running event loop". nest_asyncio patches that.
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# Reuse everything that's TTS-engine-independent from batch_dub.py
from batch_dub import (
    SAMPLE_RATE,
    VideoItem,
    build_items,
    build_output_name,
    download_video,
    extract_audio,
    fetch_metadata,
    get_video_duration,
    load_urls,
    parse_range,
    sanitize,
    strip_vocals,
)


# ============================================================
#  Defaults — male + female voice per language
# ============================================================
DEFAULT_VOICES: dict[str, dict[str, str]] = {
    "EN": {"M": "en-US-AndrewNeural",  "F": "en-US-AriaNeural"},
    "ES": {"M": "es-ES-AlvaroNeural",  "F": "es-ES-ElviraNeural"},
    "DE": {"M": "de-DE-ConradNeural",  "F": "de-DE-KatjaNeural"},
    "IT": {"M": "it-IT-DiegoNeural",   "F": "it-IT-ElsaNeural"},
    "FR": {"M": "fr-FR-HenriNeural",   "F": "fr-FR-DeniseNeural"},
    "PT": {"M": "pt-BR-AntonioNeural", "F": "pt-BR-FranciscaNeural"},
    "RU": {"M": "ru-RU-DmitryNeural",  "F": "ru-RU-SvetlanaNeural"},
    "JA": {"M": "ja-JP-KeitaNeural",   "F": "ja-JP-NanamiNeural"},
    "ZH": {"M": "zh-CN-YunxiNeural",   "F": "zh-CN-XiaoxiaoNeural"},
    "KO": {"M": "ko-KR-InJoonNeural",  "F": "ko-KR-SunHiNeural"},
    "AR": {"M": "ar-EG-ShakirNeural",  "F": "ar-EG-SalmaNeural"},
    "PL": {"M": "pl-PL-MarekNeural",   "F": "pl-PL-ZofiaNeural"},
    "TR": {"M": "tr-TR-AhmetNeural",   "F": "tr-TR-EmelNeural"},
    "NL": {"M": "nl-NL-MaartenNeural", "F": "nl-NL-FennaNeural"},
    "HU": {"M": "hu-HU-TamasNeural",   "F": "hu-HU-NoemiNeural"},
    "CS": {"M": "cs-CZ-AntoninNeural", "F": "cs-CZ-VlastaNeural"},
    "HI": {"M": "hi-IN-MadhurNeural",  "F": "hi-IN-SwaraNeural"},
}


def resolve_voice(lang: str, gender: str = "M", custom: str | None = None) -> str:
    """Pick a voice. `custom` overrides everything (must be a valid edge-tts voice name)."""
    if custom:
        return custom
    return DEFAULT_VOICES.get(lang.upper(), DEFAULT_VOICES["EN"]).get(gender.upper(), DEFAULT_VOICES["EN"]["M"])

# Max additional rate edge-tts will accept (+100% is technically allowed but
# anything over +50% sounds stressed).
MAX_RATE_PCT = 50


# ============================================================
#  TTS generation
# ============================================================
async def _tts_to_wav(
    text: str, voice: str, pitch_st: int, rate_pct: int, out_path: Path,
) -> None:
    """Generate one segment via edge-tts and save as WAV."""
    import edge_tts
    pitch_str = f"{pitch_st:+d}Hz"
    rate_str = f"{rate_pct:+d}%"
    mp3_path = out_path.with_suffix(".mp3")
    comm = edge_tts.Communicate(text, voice, pitch=pitch_str, rate=rate_str)
    await comm.save(str(mp3_path))
    # edge-tts only outputs MP3; convert to WAV (24kHz mono to match the rest)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(mp3_path),
            "-ac", "1", "-ar", str(SAMPLE_RATE),
            str(out_path),
        ],
        check=True,
    )
    mp3_path.unlink(missing_ok=True)


def _fit_segment(
    text: str, voice: str, pitch_st: int, target_dur: float, work_path: Path,
) -> tuple[np.ndarray, int, int]:
    """Generate a segment that fits within target_dur seconds.
    Returns (audio, sr, final_rate_pct)."""
    # First attempt at natural rate
    asyncio.run(_tts_to_wav(text, voice, pitch_st, 0, work_path))
    audio, sr = sf.read(work_path)
    actual = len(audio) / sr
    if actual <= target_dur * 1.02:
        return audio, sr, 0
    # Too long — compute the rate boost needed and clamp at MAX_RATE_PCT
    needed_pct = int(round((actual / target_dur - 1) * 100))
    rate_pct = min(needed_pct, MAX_RATE_PCT)
    asyncio.run(_tts_to_wav(text, voice, pitch_st, rate_pct, work_path))
    audio, sr = sf.read(work_path)
    return audio, sr, rate_pct


def _generate_tts(
    subs: list, voice: str, pitch_st: int, seg_dir: Path,
) -> tuple[list[tuple[np.ndarray, int] | None], int]:
    """Render all SRT segments. Returns (per-segment audios, sr_master)."""
    print(f"  Generating TTS (voice: {voice}, pitch: {pitch_st:+d}Hz)...", flush=True)
    results: list[tuple[np.ndarray, int] | None] = []
    n_boosted = 0
    sr_master = SAMPLE_RATE
    for i, sub in enumerate(subs):
        text = sub.content.strip()
        if not text:
            results.append(None)
            continue
        seg_path = seg_dir / f"seg_{i:04d}.wav"
        target = (sub.end - sub.start).total_seconds()
        try:
            audio, sr, rate_pct = _fit_segment(text, voice, pitch_st, target, seg_path)
            if rate_pct > 0:
                n_boosted += 1
            results.append((audio, sr))
            sr_master = sr
        except Exception as e:
            print(f"  [WARN] seg {i+1}: {e}")
            results.append(None)
    print(f"  Generated: {sum(1 for r in results if r)}/{len(subs)} (rate-boosted: {n_boosted})")
    return results, sr_master


# ============================================================
#  Master audio assembly (timeline = original SRT)
# ============================================================
def _build_master(
    subs: list,
    tts_audios: list[tuple[np.ndarray, int] | None],
    sr_master: int,
    video_duration: float,
    ambient_path: Path | None,
) -> np.ndarray:
    total_samples = int(video_duration * sr_master) + sr_master
    master = np.zeros(total_samples, dtype=np.float32)
    for sub, fit in zip(subs, tts_audios):
        if fit is None:
            continue
        audio, _ = fit
        start = int(sub.start.total_seconds() * sr_master)
        end = min(start + len(audio), len(master))
        master[start:end] += audio[:end - start].astype(np.float32)

    if ambient_path is not None:
        amb, sr_amb = sf.read(ambient_path)
        if amb.ndim > 1:
            amb = amb.mean(axis=1)
        if sr_amb != sr_master:
            from scipy import signal
            amb = signal.resample_poly(amb, sr_master, sr_amb)
        amb = amb[:len(master)] if len(amb) >= len(master) else np.pad(amb, (0, len(master) - len(amb)))
        master = master + amb.astype(np.float32) * 0.25

    peak = float(np.max(np.abs(master)))
    if peak > 0.99:
        master *= 0.99 / peak
    return master


# ============================================================
#  Per-video pipeline
# ============================================================
def dub_one(
    item: VideoItem,
    lang: str,
    voice: str,
    pitch_st: int,
    work_dir: Path,
    output_path: Path,
    remove_voice: bool,
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    seg_dir = work_dir / "segments"
    seg_dir.mkdir(exist_ok=True)

    subs = list(srt.parse(item.srt_path.read_text(encoding="utf-8")))
    print(f"  SRT: {len(subs)} segments  |  engine: edge-tts")

    video_path = work_dir / "video.mp4"
    if not video_path.exists():
        print("  Downloading video...", flush=True)
        download_video(item.url, video_path)

    ambient_path: Path | None = None
    if remove_voice:
        orig_audio = work_dir / "orig.wav"
        extract_audio(video_path, orig_audio)
        print("  Stripping vocals (Demucs)...", flush=True)
        stripped = work_dir / "ambient.wav"
        if strip_vocals(orig_audio, stripped):
            ambient_path = stripped

    tts_audios, sr_master = _generate_tts(subs, voice, pitch_st, seg_dir)
    if not any(r is not None for r in tts_audios):
        raise RuntimeError("No TTS generated for any segment")

    video_duration = get_video_duration(video_path)
    master = _build_master(subs, tts_audios, sr_master, video_duration, ambient_path)
    master_path = work_dir / "master.wav"
    sf.write(master_path, master, sr_master)

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
    shutil.rmtree(work_dir, ignore_errors=True)


# ============================================================
#  Public entry point
# ============================================================
def run_batch(
    urls,
    srt_dir,
    output_dir,
    lang: str = "EN",
    gender: str = "M",
    voice: str | None = None,
    pitch_st: int = 0,
    translate_titles: bool = False,
    remove_voice: bool = True,
    range_expr: str = "all",
    work_root: str | Path = "/tmp/sta-ru-edge",
) -> list[VideoItem]:
    """Main entry point. `voice` overrides `gender` if provided."""
    lang_uc = lang.upper()
    voice = resolve_voice(lang_uc, gender, voice)

    url_entries = load_urls(urls)
    if not url_entries:
        print("No URLs to process.")
        return []
    print(f"\n{'='*60}\nLoaded {len(url_entries)} URLs\n{'='*60}\n")

    print("Fetching metadata from YouTube...")
    items = build_items(url_entries, translate_titles=translate_titles, target_lang=lang.lower())

    srt_dir_path = Path(srt_dir)
    output_dir_path = Path(output_dir)
    for it in items:
        if it.status == "failed":
            continue
        srt_path = srt_dir_path / f"{it.n}-{lang_uc}.srt"
        if not srt_path.exists():
            it.status = "skipped"
            it.error = f"no SRT: {srt_path.name}"
            continue
        it.srt_path = srt_path
        out_name = build_output_name(it, lang, translate_titles, ext="mp4")
        it.output_path = output_dir_path / out_name

    selected = parse_range(range_expr, max(it.n for it in items) if items else 0)
    for it in items:
        if it.n not in selected and it.status == "pending":
            it.status = "skipped"
            it.error = "out of range"

    for it in items:
        if it.status == "pending" and it.output_path and it.output_path.exists():
            it.status = "done"
            it.error = "already exists"

    print(f"\n{'='*60}\nPlan\n{'='*60}")
    print(f"{'N#':>3} {'Status':>8}  Title")
    for it in items:
        title = it.title_translated if (translate_titles and it.title_translated) else it.title
        print(f"{it.n:>3} {it.status:>8}  {title[:60] or '(no metadata)'}")
    pending = [it for it in items if it.status == "pending"]
    print(f"\nPending: {len(pending)} | Done: {sum(1 for i in items if i.status == 'done')} | Skipped: {sum(1 for i in items if i.status == 'skipped')} | Failed: {sum(1 for i in items if i.status == 'failed')}")

    if not pending:
        return items

    work_root_path = Path(work_root)
    for idx, it in enumerate(pending, start=1):
        print(f"\n[{idx}/{len(pending)}] N#{it.n} — {(it.title or '')[:60]}")
        t0 = time.time()
        try:
            dub_one(
                item=it,
                lang=lang.lower(),
                voice=voice,
                pitch_st=pitch_st,
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
    ap = argparse.ArgumentParser(description="Sta-RU batch dubbing — Edge-TTS")
    ap.add_argument("--urls", required=True)
    ap.add_argument("--srt-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--lang", default="EN")
    ap.add_argument("--gender", choices=["M", "F"], default="M",
                    help="Default voice gender per language (M or F). Ignored if --voice is set.")
    ap.add_argument("--voice", default=None,
                    help="Explicit edge-tts voice name (overrides --gender). E.g. en-US-AndrewNeural")
    ap.add_argument("--pitch", type=int, default=0, help="Pitch shift in Hz, e.g. -5")
    ap.add_argument("--translate-titles", action="store_true")
    ap.add_argument("--no-remove-voice", action="store_true")
    ap.add_argument("--range", dest="range_expr", default="all")
    ap.add_argument("--work-root", default="/tmp/sta-ru-edge")
    args = ap.parse_args()

    run_batch(
        urls=args.urls,
        srt_dir=args.srt_dir,
        output_dir=args.output_dir,
        lang=args.lang,
        gender=args.gender,
        voice=args.voice,
        pitch_st=args.pitch,
        translate_titles=args.translate_titles,
        remove_voice=not args.no_remove_voice,
        range_expr=args.range_expr,
        work_root=args.work_root,
    )


if __name__ == "__main__":
    _cli()
