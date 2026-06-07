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

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it=None, **kwargs):  # type: ignore
        return it if it is not None else iter(())

# Jupyter/Colab already runs an event loop, so plain asyncio.run() raises
# "cannot be called from a running event loop". nest_asyncio patches that.
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# ANSI colour codes — Colab/Jupyter render these in cell output. Used to make
# failures in step 9 stand out in red instead of blending into normal output.
_RED = "\033[91m"
_RESET = "\033[0m"

# Reuse everything that's TTS-engine-independent from batch_dub.py.
# AMBIENT_GAIN is read via the module reference (batch_dub.AMBIENT_GAIN) so
# runtime overrides from the notebook (`batch_dub.AMBIENT_GAIN = slider.value`)
# are honoured by both classic and dynamic modes — `from batch_dub import
# AMBIENT_GAIN` would freeze the value at import time.
import batch_dub
from batch_dub import (
    MAX_VIDEO_STRETCH,
    SAMPLE_RATE,
    VideoItem,
    build_items,
    build_items_from_files,
    build_output_name,
    detect_silent_segments,
    download_video,
    extract_audio,
    fetch_metadata,
    get_video_duration,
    load_urls,
    newest_srt,
    parse_range,
    prepare_video_and_ambient,
    sanitize,
    strip_vocals,
    _build_master_dynamic,
    _build_plan_dynamic,
    _warp_video,
)


# ============================================================
#  Defaults — male + female voice per language
# ============================================================
DEFAULT_VOICES: dict[str, dict[str, str]] = {
    "EN": {"M": "en-US-AndrewNeural",  "F": "en-US-AriaNeural"},
    "ES": {"M": "es-MX-JorgeNeural",   "F": "es-MX-DaliaNeural"},
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


# Curated shortlist for the notebook voice picker: a few alternatives per
# language + gender (max 4) so the dropdown stays manageable. The first entry is
# the default and mirrors DEFAULT_VOICES. Languages not listed here fall back to
# their single DEFAULT_VOICES entry. All are free edge-tts voices.
VOICE_CHOICES: dict[str, dict[str, list[str]]] = {
    "EN": {"M": ["en-US-AndrewNeural", "en-US-BrianNeural", "en-US-GuyNeural", "en-US-ChristopherNeural"],
           "F": ["en-US-AriaNeural", "en-US-AvaNeural", "en-US-EmmaNeural", "en-US-JennyNeural"]},
    "ES": {"M": ["es-MX-JorgeNeural", "es-MX-GerardoNeural", "es-ES-AlvaroNeural", "es-ES-DarioNeural"],
           "F": ["es-MX-DaliaNeural", "es-MX-RenataNeural", "es-ES-ElviraNeural", "es-ES-XimenaNeural"]},
    "DE": {"M": ["de-DE-ConradNeural", "de-DE-BerndNeural", "de-DE-ChristophNeural", "de-DE-KillianNeural"],
           "F": ["de-DE-KatjaNeural", "de-DE-AmalaNeural", "de-DE-ElkeNeural", "de-DE-LouisaNeural"]},
    "IT": {"M": ["it-IT-DiegoNeural", "it-IT-GiuseppeNeural", "it-IT-GianniNeural", "it-IT-BenignoNeural"],
           "F": ["it-IT-ElsaNeural", "it-IT-IsabellaNeural", "it-IT-FabiolaNeural", "it-IT-ImeldaNeural"]},
    "RU": {"M": ["ru-RU-DmitryNeural"],
           "F": ["ru-RU-SvetlanaNeural", "ru-RU-DariyaNeural"]},
}


def resolve_voice(lang: str, gender: str = "M", custom: str | None = None) -> str:
    """Pick a voice. `custom` overrides everything (must be a valid edge-tts voice name)."""
    if custom:
        return custom
    return DEFAULT_VOICES.get(lang.upper(), DEFAULT_VOICES["EN"]).get(gender.upper(), DEFAULT_VOICES["EN"]["M"])

# Max additional rate edge-tts will accept (+100% is technically allowed but
# anything over +50% sounds stressed).
MAX_RATE_PCT = 50

# How much we'll slow the voice down to fill a long slot. -30% makes the voice
# noticeably more deliberate without sounding sluggish — approximates HeyGen's
# "natural pacing" behavior when the TTS is shorter than the SRT slot.
MIN_RATE_PCT = -30

# If the rendered TTS is within this fraction of the slot, leave it alone.
NATURAL_FIT_LOWER = 0.85
NATURAL_FIT_UPPER = 1.02


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
    # Retry transient edge-tts failures (502/503/504, connection drops) with
    # exponential backoff. Bing's server occasionally rate-limits a single
    # segment without hurting the rest of the batch.
    delays = [1.0, 2.0, 4.0]
    for attempt, delay in enumerate([0.0] + delays):
        if delay > 0:
            await asyncio.sleep(delay)
        try:
            comm = edge_tts.Communicate(text, voice, pitch=pitch_str, rate=rate_str)
            await comm.save(str(mp3_path))
            break
        except Exception as e:
            msg = str(e)
            transient = any(s in msg for s in ("502", "503", "504",
                                               "ServerDisconnected",
                                               "Connection reset",
                                               "Connection refused"))
            if not transient or attempt == len(delays):
                raise
            print(f"    [edge-tts] transient error ({msg[:80]}); retry "
                  f"{attempt + 1}/{len(delays)} in {delays[attempt]}s", flush=True)
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
    """Generate a segment that fits target_dur seconds.

    Strategy:
      - Too long  -> re-render with rate > 0 (speed up, capped at MAX_RATE_PCT)
      - Too short -> re-render with rate < 0 (slow down, capped at MIN_RATE_PCT)
      - Within [NATURAL_FIT_LOWER, NATURAL_FIT_UPPER] * target -> leave as-is.

    Returns (audio, sr, final_rate_pct).
    """
    asyncio.run(_tts_to_wav(text, voice, pitch_st, 0, work_path))
    audio, sr = sf.read(work_path)
    actual = len(audio) / sr
    ratio = actual / target_dur if target_dur > 0 else 1.0

    if ratio <= NATURAL_FIT_UPPER and ratio >= NATURAL_FIT_LOWER:
        return audio, sr, 0  # fits naturally

    if ratio > NATURAL_FIT_UPPER:
        # Too long -> speed up
        needed_pct = int(round((ratio - 1) * 100))
        rate_pct = min(needed_pct, MAX_RATE_PCT)
    else:
        # Too short -> slow down to fill the slot more naturally
        needed_pct = int(round((ratio - 1) * 100))      # negative
        rate_pct = max(needed_pct, MIN_RATE_PCT)

    asyncio.run(_tts_to_wav(text, voice, pitch_st, rate_pct, work_path))
    audio, sr = sf.read(work_path)
    return audio, sr, rate_pct


def _fit_segment_dd(
    text: str, voice: str, pitch_st: int, slot: float, work_path: Path,
) -> tuple[np.ndarray, int, int]:
    """For DD mode. Audio always at natural rate, except when even the maximum
    video slow-mo (MAX_VIDEO_STRETCH) wouldn't be enough to absorb a very long
    line — only then we resort to a small rate-boost.

    Slow-down is intentionally NOT applied here: DD handles short TTS by
    speeding the video up (pts < 1), so the dub stays at natural cadence.
    """
    asyncio.run(_tts_to_wav(text, voice, pitch_st, 0, work_path))
    audio, sr = sf.read(work_path)
    actual = len(audio) / sr

    cap_max = slot * MAX_VIDEO_STRETCH
    if actual > cap_max * 1.02:
        needed_pct = int(round((actual / cap_max - 1) * 100))
        rate_pct = min(needed_pct, MAX_RATE_PCT)
        asyncio.run(_tts_to_wav(text, voice, pitch_st, rate_pct, work_path))
        audio, sr = sf.read(work_path)
        return audio, sr, rate_pct

    return audio, sr, 0


def _declick(audio: np.ndarray, sr: int, ms: float = 8.0) -> np.ndarray:
    """Apply a short raised-cosine fade in/out to a TTS segment to remove the
    click heard at the start of every spoken line.

    edge-tts segments don't begin (or end) on a zero sample, so dropping one
    straight into the silent master is a step discontinuity — an audible
    high-frequency tick right when each line starts. An ~8 ms ramp removes it
    and is far too short to be perceived as a fade on speech (a phoneme lasts
    50-200 ms). The length is unchanged, so segment timing / sync is untouched.
    """
    if audio.ndim != 1:
        # edge-tts is mono, but stay safe: fade each channel along time.
        return np.stack(
            [_declick(audio[..., c], sr, ms) for c in range(audio.shape[-1])],
            axis=-1,
        )
    n = int(sr * ms / 1000.0)
    if len(audio) < 2:
        return audio
    n = min(n, len(audio) // 2)
    if n < 1:
        return audio
    ramp = (0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))).astype(np.float32)
    out = audio.astype(np.float32).copy()
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def _generate_tts(
    subs: list, voice: str, pitch_st: int, seg_dir: Path,
    dynamic_duration: bool = False,
    voice_mask: list[bool] | None = None,
    strict_tts: bool = False,
) -> tuple[list[tuple[np.ndarray, int] | None], int]:
    """Render all SRT segments. Returns (per-segment audios, sr_master).

    If dynamic_duration is True, the TTS is generated at natural rate and only
    re-rendered with a rate boost when its duration would exceed what the video
    can be slowed down to (MAX_VIDEO_STRETCH x slot). The video itself is
    stretched downstream to absorb the rest of the gap, so the dub stays natural.
    """
    label = "DD" if dynamic_duration else "rate-fit"
    print(f"  Generating TTS (voice: {voice}, pitch: {pitch_st:+d}Hz, {label})...", flush=True)
    results: list[tuple[np.ndarray, int] | None] = []
    n_sped_up = 0
    n_slowed = 0
    sr_master = SAMPLE_RATE
    for i, sub in enumerate(tqdm(subs, desc="  tts", leave=False, unit="seg")):
        text = sub.content.strip()
        if not text:
            results.append(None)
            continue
        if voice_mask is not None and not voice_mask[i]:
            results.append(None)
            continue
        seg_path = seg_dir / f"seg_{i:04d}.wav"
        slot = (sub.end - sub.start).total_seconds()
        if slot <= 0:
            results.append(None)
            continue
        try:
            if dynamic_duration:
                audio, sr, rate_pct = _fit_segment_dd(text, voice, pitch_st, slot, seg_path)
            else:
                audio, sr, rate_pct = _fit_segment(text, voice, pitch_st, slot, seg_path)
            if rate_pct > 0:
                n_sped_up += 1
            elif rate_pct < 0:
                n_slowed += 1
            audio = _declick(audio, sr)  # kill the per-segment edge tick
            results.append((audio, sr))
            sr_master = sr
        except Exception as e:
            if strict_tts:
                # A genuine synthesis failure on a needed segment: abort the
                # whole video rather than silently shipping a dub with a hole
                # in it. (Empty/silent/zero-slot segments never reach here —
                # they short-circuit to None above and are legitimate.)
                raise RuntimeError(
                    f"TTS synthesis failed at segment {i+1}/{len(subs)} "
                    f"(voice={voice}): {e}. Aborting this video so it is not "
                    f"produced incomplete."
                ) from e
            print(f"  [WARN] seg {i+1}: {e}")
            results.append(None)
    print(f"  Generated: {sum(1 for r in results if r)}/{len(subs)}  (sped up: {n_sped_up}, slowed down: {n_slowed})")
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
        master = master + batch_dub.AMBIENT_GAIN * amb.astype(np.float32)

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
    dynamic_duration: bool = False,
    cache_root: Path | None = None,
    skip_silent_segments: bool = True,
    burn_in_subs: bool = False,
    demucs_model: str = "htdemucs",
    demucs_segment: int | None = None,
    allow_no_ambient: bool = False,
    prefetched_video: Path | None = None,
    strict_tts: bool = False,
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    seg_dir = work_dir / "segments"
    seg_dir.mkdir(exist_ok=True)

    subs = list(srt.parse(item.srt_path.read_text(encoding="utf-8")))
    mode = "dynamic-duration" if dynamic_duration else "rate-fit"
    print(f"  SRT: {len(subs)} segments  |  engine: edge-tts  |  mode: {mode}")

    video_path, orig_audio, ambient_path, _vocals_path = prepare_video_and_ambient(
        item.url, work_dir, cache_root, remove_voice,
        demucs_model=demucs_model, demucs_segment=demucs_segment,
        allow_no_ambient=allow_no_ambient,
        prefetched_video=prefetched_video,
        cache_key=batch_dub.item_cache_key(item),
    )

    voice_mask: list[bool] | None = None
    if skip_silent_segments:
        voice_mask = detect_silent_segments(orig_audio, subs)
        n_skipped = sum(1 for m in voice_mask if not m)
        if n_skipped:
            print(f"  Skipping TTS for {n_skipped}/{len(subs)} segments where the original is silent")

    tts_audios, sr_master = _generate_tts(subs, voice, pitch_st, seg_dir, dynamic_duration, voice_mask, strict_tts=strict_tts)
    if not any(r is not None for r in tts_audios):
        raise RuntimeError("No TTS generated for any segment")

    video_duration = get_video_duration(video_path)
    if dynamic_duration:
        parts = _build_plan_dynamic(subs, tts_audios, video_duration, sr_master)
        final_video = _warp_video(parts, video_path, work_dir)
        master, _ = _build_master_dynamic(parts, sr_master, ambient_path)
    else:
        final_video = video_path
        master = _build_master(subs, tts_audios, sr_master, video_duration, ambient_path)

    peak = float(np.max(np.abs(master)))
    if peak > 0.99:
        master *= 0.99 / peak
    master_path = work_dir / "master.wav"
    sf.write(master_path, master, sr_master)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if burn_in_subs:
        # Burn the SRT into the video. Requires re-encoding (can't -c:v copy
        # with a video filter). Use HEVC to keep file sizes reasonable.
        # ffmpeg's 'subtitles' filter needs the file path as a single arg;
        # escape characters that would be interpreted by the filter.
        srt_arg = str(item.srt_path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        mux_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(final_video),
            "-i", str(master_path),
            "-filter_complex", f"[0:v]subtitles='{srt_arg}'[v]",
            "-map", "[v]", "-map", "1:a:0",
            "-c:a", "aac", "-b:a", "192k",
            "-c:v", "hevc_nvenc" if _has_nvenc_local() else "libx265",
        ]
        if _has_nvenc_local():
            mux_cmd += ["-preset", "p5", "-rc", "vbr", "-cq", "33", "-b:v", "0", "-tag:v", "hvc1"]
        else:
            mux_cmd += ["-preset", "medium", "-crf", "28", "-tag:v", "hvc1",
                        "-x265-params", "log-level=error"]
        mux_cmd += ["-pix_fmt", "yuv420p", "-shortest", str(output_path)]
        subprocess.run(mux_cmd, check=True)
    else:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(final_video),
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


def dub_one_split_aware(
    item: VideoItem,
    lang: str,
    voice: str,
    pitch_st: int,
    work_dir: Path,
    output_path: Path,
    remove_voice: bool,
    dynamic_duration: bool = False,
    cache_root: Path | None = None,
    skip_silent_segments: bool = True,
    burn_in_subs: bool = False,
    demucs_model: str = "htdemucs",
    demucs_segment: int | None = None,
    allow_no_ambient: bool = False,
    max_part_seconds: int | None = None,
    normalize_local: bool = False,
    strict_tts: bool = False,
) -> None:
    """Wrap dub_one with a pre-pipeline split for long videos.

    If the source is <= max_part_seconds (or splitting is disabled), this is
    just dub_one passing the pre-downloaded video through. If it's longer,
    we cut the video and SRT at the best silences so every sub-part fits, run
    the normal pipeline on each, and concat the dubbed outputs.

    Source can be a YouTube URL (item.url) or a local file (item.source_path,
    set for uploaded / Drive videos). A local source skips yt-dlp entirely —
    that's the path that dodges YouTube's cloud-IP bot wall.
    """
    import dataclasses
    import split_pipeline

    work_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: get the source video on disk. Use cache when available so the
    # split-aware path doesn't re-download (or re-copy) on every run.
    download_dir = work_dir / "_download"
    download_dir.mkdir(exist_ok=True)
    full_video = download_dir / "video.mp4"
    cache_dir = (cache_root / batch_dub.item_cache_key(item)) if cache_root else None
    cached_video = cache_dir / "video.mp4" if cache_dir else None
    if cached_video and cached_video.exists():
        shutil.copy(cached_video, full_video)
        print("  Video from cache")
    elif item.source_path is not None:
        src = Path(item.source_path)
        if not src.exists():
            raise RuntimeError(f"source video not found: {src}")
        if normalize_local:
            print("  Normalizing local video (CFR re-encode)...", flush=True)
            batch_dub.normalize_local_video(src, full_video)
        else:
            print(f"  Using local video: {src.name}", flush=True)
            shutil.copy(src, full_video)
        if cached_video:
            cache_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(full_video, cached_video)
    else:
        print("  Downloading video...", flush=True)
        batch_dub.download_video(item.url, full_video)
        if cached_video:
            cache_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(full_video, cached_video)

    duration = batch_dub.get_video_duration(full_video)

    # Short path: one go, normal pipeline, just skip the second download.
    if max_part_seconds is None or duration <= max_part_seconds:
        dub_one(
            item=item, lang=lang, voice=voice, pitch_st=pitch_st,
            work_dir=work_dir / "whole", output_path=output_path,
            remove_voice=remove_voice, dynamic_duration=dynamic_duration,
            cache_root=cache_root,
            skip_silent_segments=skip_silent_segments,
            burn_in_subs=burn_in_subs,
            demucs_model=demucs_model, demucs_segment=demucs_segment,
            allow_no_ambient=allow_no_ambient,
            prefetched_video=full_video,
            strict_tts=strict_tts,
        )
        shutil.rmtree(download_dir, ignore_errors=True)
        return

    # Long path: split at silences, dub each part, concat at the end.
    print(f"  Long video ({duration:.0f}s > {max_part_seconds}s). "
          f"Planning silence-aware split...", flush=True)
    cuts = split_pipeline.plan_cuts(
        full_video, max_part_seconds, srt_path=item.srt_path,
    )
    parts_dir = work_dir / "_parts"
    parts_dir.mkdir(exist_ok=True)
    # split_video may shift cuts to the next keyframe — use the actual cuts
    # (not the planned ones) for the SRT so captions stay aligned.
    video_parts, actual_cuts = split_pipeline.split_video(
        full_video, cuts, parts_dir,
    )
    srt_parts = split_pipeline.split_srt(
        item.srt_path, actual_cuts, parts_dir, video_duration=duration,
    )
    if len(video_parts) != len(srt_parts):
        raise RuntimeError(
            f"split produced {len(video_parts)} video parts but "
            f"{len(srt_parts)} SRT parts; this is a bug."
        )
    print(f"  Split into {len(video_parts)} parts at "
          f"{[f'{c:.1f}s' for c in actual_cuts]} (planned: "
          f"{[f'{c:.1f}s' for c in cuts]})", flush=True)

    dubbed_parts: list[Path] = []
    for i, (vp_path, sp_path) in enumerate(zip(video_parts, srt_parts), start=1):
        print(f"\n  --- Part {i}/{len(video_parts)} ---", flush=True)
        sub_item = dataclasses.replace(item, srt_path=sp_path)
        sub_work = work_dir / f"part_{i:03d}"
        sub_output = parts_dir / f"dubbed_{i:03d}.mp4"
        # cache_root=None for sub-parts: their url_hash would collide with
        # the full video's cache and pollute it with per-part artifacts.
        dub_one(
            item=sub_item, lang=lang, voice=voice, pitch_st=pitch_st,
            work_dir=sub_work, output_path=sub_output,
            remove_voice=remove_voice, dynamic_duration=dynamic_duration,
            cache_root=None,
            skip_silent_segments=skip_silent_segments,
            burn_in_subs=burn_in_subs,
            demucs_model=demucs_model, demucs_segment=demucs_segment,
            allow_no_ambient=allow_no_ambient,
            prefetched_video=vp_path,
            strict_tts=strict_tts,
        )
        dubbed_parts.append(sub_output)

    print(f"\n  Concatenating {len(dubbed_parts)} dubbed parts...", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = split_pipeline.concat_dubbed_parts(dubbed_parts, output_path)
    if not ok:
        raise RuntimeError("concat of dubbed parts failed")
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  ✓ {output_path.name} ({size_mb:.1f} MB)  [from {len(dubbed_parts)} parts]")
    shutil.rmtree(work_dir, ignore_errors=True)


def _has_nvenc_local() -> bool:
    """Local proxy so we don't have to import _has_nvenc through batch_dub."""
    from batch_dub import _has_nvenc
    return _has_nvenc()


# ============================================================
#  Public entry point
# ============================================================
def run_batch(
    urls=None,
    srt_dir=None,
    output_dir=None,
    lang: str = "EN",
    gender: str = "M",
    voice: str | None = None,
    pitch_st: int = 0,
    translate_titles: bool = False,
    remove_voice: bool = True,
    range_expr: str = "all",
    work_root: str | Path = "/tmp/sta-ru-edge",
    dynamic_duration: bool = False,
    cache_root: str | Path | None = "/tmp/sta-ru-cache",
    skip_silent_segments: bool = True,
    burn_in_subs: bool = False,
    demucs_model: str = "htdemucs",
    demucs_segment: int | None = None,
    allow_no_ambient: bool = False,
    max_part_seconds: int | None = 35 * 60,
    source_paths: list[str | Path] | None = None,
    normalize_local: bool = False,
    strict_tts: bool = False,
) -> list[VideoItem]:
    """Main entry point. `voice` overrides `gender` if provided.
    `max_part_seconds`: split videos longer than this at silences so each
    sub-part runs through the dub pipeline as a normal short video; default
    is 35 min. Pass None to disable splitting.

    Source selection:
      - `source_paths` set -> dub local files (uploaded / Drive). No yt-dlp, no
        metadata, no YouTube bot wall. Output is named after each file's stem.
      - otherwise -> `urls` are YouTube links, as before.

    Subtitle pairing:
      - exactly one video -> use the newest .srt in srt_dir (the last one the
        user uploaded), regardless of its filename.
      - more than one     -> pair each by the numbered convention {N#}-{LANG}.srt.

    `normalize_local`: re-encode local files to constant-frame-rate MP4 first
    (opt-in; protects against VFR/.mkv drift)."""
    lang_uc = lang.upper()
    voice = resolve_voice(lang_uc, gender, voice)

    using_files = bool(source_paths)
    if using_files:
        items = build_items_from_files(list(source_paths))
        print(f"\n{'='*60}\nLoaded {len(items)} local video file(s)\n{'='*60}\n")
    else:
        url_entries = load_urls(urls)
        if not url_entries:
            print("No URLs to process.")
            return []
        print(f"\n{'='*60}\nLoaded {len(url_entries)} URLs\n{'='*60}\n")
        print("Fetching metadata from YouTube...")
        items = build_items(url_entries, translate_titles=translate_titles, target_lang=lang.lower())

    srt_dir_path = Path(srt_dir)
    output_dir_path = Path(output_dir)

    # Subtitle pairing rule. A single video shouldn't force the user to rename
    # their SRT to {N#}-{LANG}.srt, so we take the newest SRT in the dir. With
    # several videos we need the numbered convention to tell them apart.
    single = sum(1 for it in items if it.status != "failed") == 1
    auto_srt = newest_srt(srt_dir_path, lang_uc) if single else None
    if single and auto_srt is not None:
        print(f"  Single video -> using newest subtitle: {auto_srt.name}")

    for it in items:
        if it.status == "failed":
            continue
        srt_path = auto_srt if single else (srt_dir_path / f"{it.n}-{lang_uc}.srt")
        if srt_path is None or not srt_path.exists():
            it.status = "skipped"
            it.error = ("no SRT in " + str(srt_dir_path) if single
                        else f"no SRT: {it.n}-{lang_uc}.srt")
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
            dub_one_split_aware(
                item=it,
                lang=lang.lower(),
                voice=voice,
                pitch_st=pitch_st,
                work_dir=work_root_path / f"n{it.n}",
                output_path=it.output_path,
                remove_voice=remove_voice,
                dynamic_duration=dynamic_duration,
                cache_root=Path(cache_root) if cache_root else None,
                skip_silent_segments=skip_silent_segments,
                burn_in_subs=burn_in_subs,
                demucs_model=demucs_model,
                demucs_segment=demucs_segment,
                allow_no_ambient=allow_no_ambient,
                max_part_seconds=max_part_seconds,
                normalize_local=normalize_local,
                strict_tts=strict_tts,
            )
            it.status = "done"
            print(f"  Elapsed: {time.time() - t0:.1f}s")
        except Exception as e:
            it.status = "failed"
            it.error = str(e)
            print(f"{_RED}  ✗ FAILED: {e}{_RESET}")

    print(f"\n{'='*60}\nSummary\n{'='*60}")
    for s in ("done", "skipped", "failed"):
        ns = [it.n for it in items if it.status == s]
        if ns:
            line = f"  {s:>8} ({len(ns)}): {ns}"
            print(f"{_RED}{line}{_RESET}" if s == "failed" else line)
    print(f"\nOutputs in: {output_dir_path}")
    return items


def run_batch_multilang(
    urls=None,
    srt_dir=None,
    output_dir=None,
    primary_lang: str = "EN",
    secondary_langs: list[str] | None = None,
    gender: str = "M",
    voice: str | None = None,
    pitch_st: int = 0,
    translate_titles: bool = False,
    remove_voice: bool = True,
    range_expr: str = "all",
    work_root: str | Path = "/tmp/sta-ru-edge",
    dynamic_duration: bool = False,
    cache_root: str | Path | None = "/tmp/sta-ru-cache",
    skip_silent_segments: bool = True,
    burn_in_subs: bool = False,
    demucs_model: str = "htdemucs",
    demucs_segment: int | None = None,
    allow_no_ambient: bool = False,
    max_part_seconds: int | None = 35 * 60,
    source_paths: list[str | Path] | None = None,
    normalize_local: bool = False,
    strict_tts: bool = True,
    youtube_dead_threshold: int = 2,
) -> list[VideoItem]:
    """Breadth-first, multi-language, YouTube-resilient batch.

    The expensive YouTube-dependent work (download + Demucs ambient) is done
    ONCE per video and cached under {cache_root}/{key}/. Producing the same
    video in another language reuses that cache — no yt-dlp, no Demucs, just
    TTS + mux. This function exploits that to squeeze the most out of a Colab
    session even when YouTube's cloud-IP bot wall kills downloads mid-run.

    Order (breadth-first — the primary language is the priority):
      1. PRIMARY pass over every video, in `primary_lang` with the configured
         voice. If downloads start hitting the bot wall on
         `youtube_dead_threshold` consecutive videos, stop downloading; the
         remaining un-cached videos are left 'skipped'.
      2. SECONDARY passes, one per language in `secondary_langs` (in order),
         each dubbing ONLY the already-cached videos with that language's
         DEFAULT voice and the same options. Videos that never downloaded are
         skipped (can't fetch them without YouTube).

    So: if YouTube dies after N videos, you still get primary+secondaries for
    those N; and if Colab itself dies, whatever finished is intact. Secondary
    languages are pure bonus and never block the primary.

    `strict_tts=True`: a genuine TTS synthesis failure aborts that video (no
    partial dub) rather than warning and leaving a silent gap.

    Returns a flat list of VideoItem (one record per video+language attempted),
    each with its own status/output_path — drop-in for the download cell.
    """
    import dataclasses

    primary = primary_lang.upper()
    secondaries: list[str] = []
    for lg in (secondary_langs or []):
        u = (lg or "").upper()
        if u and u != primary and u not in secondaries:
            secondaries.append(u)

    cache_root_p = Path(cache_root) if cache_root else None
    work_root_path = Path(work_root)
    srt_dir_path = Path(srt_dir)

    # Output base: if output_dir already ends in the primary lang (…/Dubbing/EN),
    # use its parent so every language lands in …/Dubbing/{LANG}.
    out_base = Path(output_dir)
    if out_base.name.upper() == primary:
        out_base = out_base.parent

    def out_dir_for(lang: str) -> Path:
        return out_base / lang.upper()

    # Build items ONCE (single metadata sweep) and reuse for every language, so
    # the secondary passes never call YouTube again (real titles preserved).
    using_files = bool(source_paths)
    if using_files:
        items = build_items_from_files(list(source_paths))
        print(f"\n{'='*60}\nLoaded {len(items)} local video file(s)\n{'='*60}\n")
    else:
        url_entries = load_urls(urls)
        if not url_entries:
            print("No URLs to process.")
            return []
        print(f"\n{'='*60}\nLoaded {len(url_entries)} URLs\n{'='*60}\n")
        print("Fetching metadata from YouTube...")
        items = build_items(url_entries, translate_titles=translate_titles,
                            target_lang=primary.lower())

    selected = parse_range(range_expr, max((it.n for it in items), default=0))
    single = sum(1 for it in items if it.status != "failed") == 1

    primary_voice = resolve_voice(primary, gender, voice)
    print(f"\nPrimary:   {primary}  (voice: {primary_voice})")
    if secondaries:
        print("Secondary: " + ", ".join(
            f"{lg}->{resolve_voice(lg, gender, None)}" for lg in secondaries)
            + "   (default voice; cached videos only)")
    else:
        print("Secondary: (none)")
    print(f"strict_tts={strict_tts}  |  youtube_dead_threshold={youtube_dead_threshold}")

    def pair_srt(item: VideoItem, lang: str):
        # Single-video + primary-only run keeps the 'newest SRT wins'
        # convenience. Anything multi-video or multi-language needs the
        # numbered {N#}-{LANG}.srt convention to tell files apart.
        if single and not secondaries:
            p = newest_srt(srt_dir_path, lang)
            if p is not None:
                return p
        p = srt_dir_path / f"{item.n}-{lang.upper()}.srt"
        return p if p.exists() else None

    def video_cached(item: VideoItem) -> bool:
        if not cache_root_p:
            return False
        return (cache_root_p / batch_dub.item_cache_key(item) / "video.mp4").exists()

    def run_one(item: VideoItem, lang: str, voice_name: str, idx: int, total: int):
        """Process one (video, language). Returns a VideoItem result record."""
        rec = dataclasses.replace(item)
        out_path = out_dir_for(lang) / build_output_name(item, lang, translate_titles, ext="mp4")
        rec.output_path = out_path
        if out_path.exists():
            rec.status, rec.error = "done", "already exists"
            print(f"  (skip) N#{item.n} [{lang}] already exists")
            return rec
        srt_path = pair_srt(item, lang)
        if srt_path is None:
            rec.status, rec.error = "skipped", f"no SRT: {item.n}-{lang.upper()}.srt"
            return rec
        item.srt_path = rec.srt_path = srt_path
        print(f"\n[{idx}/{total}] N#{item.n} [{lang}] — {(item.title or '')[:55]}")
        t0 = time.time()
        dub_one_split_aware(
            item=item, lang=lang.lower(), voice=voice_name, pitch_st=pitch_st,
            work_dir=work_root_path / f"n{item.n}_{lang.upper()}",
            output_path=out_path,
            remove_voice=remove_voice, dynamic_duration=dynamic_duration,
            cache_root=cache_root_p,
            skip_silent_segments=skip_silent_segments,
            burn_in_subs=burn_in_subs,
            demucs_model=demucs_model, demucs_segment=demucs_segment,
            allow_no_ambient=allow_no_ambient,
            max_part_seconds=max_part_seconds,
            normalize_local=normalize_local,
            strict_tts=strict_tts,
        )
        rec.status = "done"
        print(f"  Elapsed: {time.time() - t0:.1f}s")
        return rec

    all_results: list[VideoItem] = []
    youtube_dead = False
    consecutive_botwall = 0

    # ---------- PRIMARY PASS ----------
    print(f"\n{'='*60}\nPRIMARY pass — {primary}\n{'='*60}")
    pending = [it for it in items if it.status != "failed" and it.n in selected]
    for i, it in enumerate(pending, start=1):
        if youtube_dead and not video_cached(it):
            rec = dataclasses.replace(it)
            rec.output_path = out_dir_for(primary) / build_output_name(it, primary, translate_titles, ext="mp4")
            rec.status, rec.error = "skipped", "youtube blocked (download halted)"
            all_results.append(rec)
            continue
        try:
            rec = run_one(it, primary, primary_voice, i, len(pending))
            if rec.status == "done":
                consecutive_botwall = 0
            all_results.append(rec)
        except Exception as e:
            rec = dataclasses.replace(it)
            rec.output_path = out_dir_for(primary) / build_output_name(it, primary, translate_titles, ext="mp4")
            rec.status, rec.error = "failed", str(e)
            print(f"{_RED}  ✗ FAILED [{primary}] N#{it.n}: {e}{_RESET}")
            all_results.append(rec)
            # Only a download-stage bot wall (video not cached) counts toward
            # 'YouTube is dead'. A strict_tts abort or a cached-video error
            # never trips it.
            if (not video_cached(it)) and batch_dub._is_botcheck(str(e)):
                consecutive_botwall += 1
                if consecutive_botwall >= youtube_dead_threshold and not youtube_dead:
                    youtube_dead = True
                    print(f"{_RED}  ⚠ YouTube appears blocked after "
                          f"{consecutive_botwall} consecutive bot-wall failures. "
                          f"Halting downloads — will finish CACHED videos in the "
                          f"other languages.{_RESET}")

    # ---------- SECONDARY PASSES (cache-gated, no YouTube) ----------
    for lang in secondaries:
        lang_voice = resolve_voice(lang, gender, None)
        cached_items = [it for it in items
                        if it.status != "failed" and it.n in selected and video_cached(it)]
        print(f"\n{'='*60}\nSECONDARY pass — {lang}  (voice {lang_voice}; "
              f"{len(cached_items)} cached video(s))\n{'='*60}")
        if not cached_items:
            print("  No cached videos available — nothing to do in this language.")
            continue
        if translate_titles:
            batch_dub._translate_titles_inplace(items, lang.lower())
        for i, it in enumerate(cached_items, start=1):
            try:
                all_results.append(run_one(it, lang, lang_voice, i, len(cached_items)))
            except Exception as e:
                rec = dataclasses.replace(it)
                rec.output_path = out_dir_for(lang) / build_output_name(it, lang, translate_titles, ext="mp4")
                rec.status, rec.error = "failed", str(e)
                print(f"{_RED}  ✗ FAILED [{lang}] N#{it.n}: {e}{_RESET}")
                all_results.append(rec)

    # ---------- SUMMARY ----------
    print(f"\n{'='*60}\nSummary (by language)\n{'='*60}")

    def _lang_of(r: VideoItem) -> str:
        return r.output_path.parent.name.upper() if r.output_path else "?"

    for lg in [primary] + secondaries:
        grp = [r for r in all_results if _lang_of(r) == lg]
        done = sorted(r.n for r in grp if r.status == "done")
        skip = sorted(r.n for r in grp if r.status == "skipped")
        fail = sorted(r.n for r in grp if r.status == "failed")
        line = f"  {lg}:  done={len(done)} {done}  |  skipped={len(skip)} {skip}  |  failed={len(fail)} {fail}"
        print(f"{_RED}{line}{_RESET}" if fail else line)
    if youtube_dead:
        print(f"\n{_RED}YouTube blocked mid-run: un-cached videos were left "
              f"un-downloaded; cached ones were completed in every requested "
              f"language.{_RESET}")
    print(f"\nOutputs base: {out_base}")
    return all_results


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
    ap.add_argument("--dynamic-duration", action="store_true",
                    help="Stretch the video to fit the dub (slower; voice stays at natural rate)")
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
        dynamic_duration=args.dynamic_duration,
    )


if __name__ == "__main__":
    _cli()
