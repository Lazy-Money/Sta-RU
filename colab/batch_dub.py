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
import hashlib
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
import soundfile as sf
import srt

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it=None, **kwargs):  # type: ignore
        return it if it is not None else iter(())

# Heavy deps (torch, TTS, pyrubberband) are imported lazily inside the
# functions that need them, so batch_dub_edge can reuse the lightweight
# helpers (load_urls, build_items, download_video, ...) without forcing
# the Edge-TTS notebook to install XTTS's dependency stack.


# ============================================================
#  Constants
# ============================================================
MAX_SPEED_RATIO = 1.4         # legacy mode: max audio compression
SAMPLE_RATE = 24000

# Dynamic duration mode: how far we're allowed to stretch the video
# to accommodate a longer dubbed audio.
MAX_VIDEO_STRETCH = 1.5       # video can be slowed down up to 50% (plays at 1/1.5x)
MIN_VIDEO_PTS = 0.7           # video can be sped up up to ~1.43x (plays at 1/0.7x)
MAX_AUDIO_COMPRESS = 1.4      # if video stretch is not enough, audio can also speed up to 1.4x

# Ambient (no-vocals stem) gain when mixing under the TTS.
# Demucs delivers the ambient at roughly the same level it had in the source
# mix, which sits below the speaker. Boosting it ~+5 dB makes workshop noise,
# music and room tone audible without the listener having to crank the player.
# Final master is peak-normalised, so this only shifts the TTS/ambient ratio.
AMBIENT_GAIN = 1.8

# If the extracted ambient stem comes back quieter than this RMS, we assume
# Demucs over-stripped (treated workshop noise / music as vocals). The dub
# still renders, but we shout in the log so it's obvious without listening.
AMBIENT_QUIET_RMS = 0.01

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
_NUM_URL_PREFIX = re.compile(r"^\s*(\d+)\s*[:=>\-]\s*(https?://\S+)\s*$")


def load_urls(source: str | Path | list[str]) -> list[tuple[int | None, str]]:
    """Load URL entries from a list, .txt or .csv file, or a multiline string.

    Each entry is returned as (n_override, url). If a line uses the
    'N#: URL' / 'N#= URL' / 'N#- URL' / 'N#> URL' form, n_override is set
    to that number; otherwise it's None and the caller assigns one
    sequentially.
    """
    if isinstance(source, list):
        lines = source
    elif isinstance(source, (str, Path)) and Path(source).is_file():
        path = Path(source)
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                # Treat anything that looks like a URL as data
                lines = []
                for row in reader:
                    if row and row[0].strip():
                        lines.append(row[0].strip())
        else:
            lines = path.read_text(encoding="utf-8").splitlines()
    else:
        lines = str(source).splitlines()

    entries: list[tuple[int | None, str]] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _NUM_URL_PREFIX.match(line)
        if m:
            entries.append((int(m.group(1)), m.group(2)))
        elif line.startswith("http"):
            entries.append((None, line))
        else:
            print(f"[WARN] skipping non-URL line: {line[:60]}")
    return entries


def load_url_strings(source) -> list[str]:
    """Convenience helper: just the URLs (drops any N# override). Kept for callers
    that only need a list of URLs, e.g. the preview cell."""
    return [u for _, u in load_urls(source)]


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


def build_items(
    entries: list[tuple[int | None, str]] | list[str],
    translate_titles: bool,
    target_lang: str,
) -> list[VideoItem]:
    """Build VideoItem list with metadata for each URL. Metadata failure is non-fatal:
    the item keeps status 'pending' with an empty title; output naming falls back to
    {N#}-{LANG}.mp4.

    `entries` is either a list of (n_override, url) tuples (output of load_urls) or
    a plain list of URLs (back-compat)."""
    items: list[VideoItem] = []
    auto_n = 0
    used_ns: set[int] = set()
    for entry in entries:
        if isinstance(entry, tuple):
            n_override, url = entry
        else:
            n_override, url = None, entry
        if n_override is not None:
            n = n_override
        else:
            auto_n += 1
            while auto_n in used_ns:
                auto_n += 1
            n = auto_n
        if n in used_ns:
            print(f"[WARN] duplicate N#{n}; will overwrite output if not skipped")
        used_ns.add(n)

        meta = fetch_metadata(url)
        item = VideoItem(n=n, url=url)
        if "_error" in meta:
            item.error = f"metadata unavailable: {meta['_error'][:120]}"
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
    """Build output filename.
    Full form  : '{date} - {N#} - {title}-{LANG}.{ext}'
    Fallback   : '{N#}-{LANG}.{ext}' (when YouTube metadata is unavailable)"""
    base_title = item.title_translated if (translate and item.title_translated) else item.title
    if not base_title:
        # No metadata at all -> use SRT-style minimal name
        return f"{item.n}-{lang.upper()}.{ext}"
    base_title = sanitize(base_title)
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


# ============================================================
#  Voice Activity Detection
# ============================================================
def detect_silent_segments(
    orig_audio_path: Path, subs: list, threshold_db: float = -38.0
) -> list[bool]:
    """For each SRT segment, return True if the original speaker is actually
    audible in that time range. Skipping TTS where the original was silent
    avoids the dub talking over moments the speaker spent thinking / gesturing.

    The check is per-segment RMS in dB. -38 dB is permissive enough to allow
    quiet whispered speech but suppress most pure-ambient stretches.
    """
    try:
        audio, sr = sf.read(orig_audio_path)
    except Exception:
        return [True] * len(subs)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    result: list[bool] = []
    for sub in subs:
        s = int(sub.start.total_seconds() * sr)
        e = int(sub.end.total_seconds() * sr)
        chunk = audio[s:e]
        if len(chunk) < sr // 20:  # < 50ms
            result.append(True)
            continue
        rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
        if rms < 1e-6:
            result.append(False)
            continue
        db = 20.0 * np.log10(rms)
        result.append(db > threshold_db)
    return result


# ============================================================
#  Per-URL cache (cross-language reuse)
# ============================================================
def url_cache_key(url: str) -> str:
    """Stable short key per YouTube URL — same across languages."""
    return hashlib.md5(url.encode("utf-8")).hexdigest()[:12]


def _log_ambient_stats(path: Path) -> None:
    """Print dur/rms/peak of the ambient stem so over-stripping is visible
    in the log. Demucs sometimes throws away non-musical ambient (workshop
    tools, motors) because they look like vocals to a model trained on pop."""
    try:
        a, sr = sf.read(path)
    except Exception as e:
        print(f"  [WARN] could not read ambient.wav: {e}")
        return
    if a.ndim > 1:
        a = a.mean(axis=1)
    if a.size == 0:
        print("  [WARN] ambient.wav is empty")
        return
    rms = float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(a)))
    line = f"  Ambient: dur={len(a)/sr:.1f}s  rms={rms:.5f}  peak={peak:.3f}"
    if rms < AMBIENT_QUIET_RMS:
        line += (f"  [WARN] very quiet (< {AMBIENT_QUIET_RMS}) — "
                 f"Demucs likely over-stripped; try another model "
                 f"(htdemucs_ft, mdx_extra) or re-extract")
    print(line)


def prepare_video_and_ambient(
    url: str,
    work_dir: Path,
    cache_root: Path | None,
    remove_voice: bool,
    demucs_model: str = "htdemucs",
    demucs_segment: int | None = None,
) -> tuple[Path, Path, Path | None, Path | None]:
    """Download the video (or fetch from cache), extract its mono audio, and
    optionally produce the ambient AND isolated-vocals tracks. When
    `cache_root` is provided, every artifact is kept under
    `{cache_root}/{url_hash}/` so subsequent runs in other languages reuse them.
    Returns (video_path, orig_audio_path, ambient_path_or_None, vocals_path_or_None)."""
    video_path = work_dir / "video.mp4"
    orig_audio = work_dir / "orig.wav"
    cache_dir = (cache_root / url_cache_key(url)) if cache_root else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    cached_video = cache_dir / "video.mp4" if cache_dir else None
    cached_orig = cache_dir / "orig.wav" if cache_dir else None
    cached_ambient = cache_dir / "ambient.wav" if cache_dir else None
    cached_vocals = cache_dir / "vocals.wav" if cache_dir else None

    # Video
    if cached_video and cached_video.exists():
        shutil.copy(cached_video, video_path)
        print("  Video from cache")
    else:
        print("  Downloading video...", flush=True)
        download_video(url, video_path)
        if cached_video:
            shutil.copy(video_path, cached_video)

    # Original mono audio (needed for voice ref by XTTS and for VAD)
    if cached_orig and cached_orig.exists():
        shutil.copy(cached_orig, orig_audio)
    else:
        extract_audio(video_path, orig_audio)
        if cached_orig:
            shutil.copy(orig_audio, cached_orig)

    # Ambient + isolated vocals (optional)
    ambient_path: Path | None = None
    vocals_path: Path | None = None
    if remove_voice:
        stripped = work_dir / "ambient.wav"
        vocals = work_dir / "vocals.wav"
        cache_hit = (
            cached_ambient and cached_ambient.exists()
            and (not cached_vocals or cached_vocals.exists())
        )
        if cache_hit:
            shutil.copy(cached_ambient, stripped)
            ambient_path = stripped
            if cached_vocals and cached_vocals.exists():
                shutil.copy(cached_vocals, vocals)
                vocals_path = vocals
            print("  Ambient & vocals from cache")
        else:
            print(f"  Stripping vocals (Demucs: {demucs_model})...", flush=True)
            if strip_vocals(orig_audio, stripped, vocals_out=vocals,
                            demucs_model=demucs_model, demucs_segment=demucs_segment):
                ambient_path = stripped
                if cached_ambient:
                    shutil.copy(stripped, cached_ambient)
                if vocals.exists():
                    vocals_path = vocals
                    if cached_vocals:
                        shutil.copy(vocals, cached_vocals)
        if ambient_path is not None:
            _log_ambient_stats(ambient_path)
    return video_path, orig_audio, ambient_path, vocals_path


def get_video_duration(video_path: Path) -> float:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(probe.stdout.strip())


def get_video_fps(video_path: Path) -> float:
    """Read the source video's frame rate via ffprobe. Falls back to 30 if unparseable."""
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
            ],
            capture_output=True, text=True, check=True,
        )
        rate = r.stdout.strip()
        if "/" in rate:
            num, denom = rate.split("/", 1)
            fps = float(num) / float(denom) if float(denom) else 0.0
        else:
            fps = float(rate)
        if 1 < fps < 240:
            return fps
    except Exception:
        pass
    return 30.0


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
def strip_vocals(
    audio_path: Path, out_path: Path, vocals_out: Path | None = None,
    demucs_model: str = "htdemucs", demucs_segment: int | None = None,
) -> bool:
    """Run Demucs to separate vocals from the rest.

    Writes the no-vocals stem (ambient) to `out_path`. If `vocals_out` is
    provided, also copies the isolated vocals stem there — useful as a clean
    voice-clone reference for XTTS.

    `demucs_model` picks the Demucs model (e.g. "htdemucs", "hdemucs_mmi",
    "mdx_extra_q"). Lighter models cut RAM usage at some quality cost.
    `demucs_segment` (seconds) caps the chunk length to further reduce RAM —
    leave as None for the model default.

    Returns True on success, False if Demucs isn't installed or fails.
    """
    try:
        import demucs.separate  # noqa: F401
    except ImportError:
        print("[WARN] demucs not installed; skipping vocal removal")
        return False
    work = audio_path.parent / "demucs_out"
    work.mkdir(exist_ok=True)
    cmd = [
        sys.executable, "-m", "demucs.separate",
        "--two-stems", "vocals",
        "-n", demucs_model,
        "-o", str(work),
    ]
    if demucs_segment is not None:
        cmd += ["--segment", str(demucs_segment)]
    cmd.append(str(audio_path))
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"[WARN] demucs failed: {e.stderr.decode()[-200:] if e.stderr else e}")
        return False
    no_vocals = next(work.rglob("no_vocals.wav"), None)
    if not no_vocals:
        return False
    shutil.copy(no_vocals, out_path)
    if vocals_out is not None:
        vocals = next(work.rglob("vocals.wav"), None)
        if vocals:
            shutil.copy(vocals, vocals_out)
    shutil.rmtree(work, ignore_errors=True)
    return True


# ============================================================
#  TTS + time-fit
# ============================================================
def time_fit(audio: np.ndarray, sr: int, target_duration: float) -> tuple[np.ndarray, bool]:
    """Compress audio to fit target_duration (no expansion). Returns (audio, capped)."""
    import pyrubberband as pyrb
    actual = len(audio) / sr
    if actual <= target_duration * 1.02:
        return audio, False
    speed = actual / target_duration
    capped = speed > MAX_SPEED_RATIO
    speed = min(speed, MAX_SPEED_RATIO)
    return pyrb.time_stretch(audio, sr, speed), capped


def _generate_tts_raw(
    subs: list, tts: TTS, lang: str, ref_path: Path, seg_dir: Path,
    hq_mode: bool = False,
    voice_mask: list[bool] | None = None,
) -> list[tuple[np.ndarray, int] | None]:
    """Generate TTS per segment at natural speed (no time-fit).
    hq_mode enables beam search + lower temperature for higher quality (slower).
    voice_mask[i] False => skip that segment (original speaker was silent)."""
    tts_kwargs: dict = {}
    if hq_mode:
        # Conservative HQ: only nudge temperature down from the 0.75 default.
        # Other knobs (num_beams, top_k, top_p) reliably trigger XTTS-v2's
        # decoder shape bug, so we keep them at defaults.
        tts_kwargs = {"temperature": 0.7}
    print(f"  Generating TTS{' (HQ mode)' if hq_mode else ''}...", flush=True)
    raw: list[tuple[np.ndarray, int] | None] = []
    for i, sub in enumerate(tqdm(subs, desc="  tts", leave=False, unit="seg")):
        text = sub.content.strip()
        if not text:
            raw.append(None)
            continue
        if voice_mask is not None and not voice_mask[i]:
            raw.append(None)
            continue
        seg_path = seg_dir / f"seg_{i:04d}.wav"
        try:
            tts.tts_to_file(
                text=text,
                file_path=str(seg_path),
                speaker_wav=str(ref_path),
                language=lang,
                split_sentences=False,
                **tts_kwargs,
            )
            audio, sr_seg = sf.read(seg_path)
            raw.append((audio, sr_seg))
        except Exception as e:
            print(f"  [WARN] seg {i+1}: {e}")
            raw.append(None)
    print(f"  Generated: {sum(1 for r in raw if r)}/{len(subs)}")
    return raw


def _build_master_classic(
    subs: list,
    raw_tts: list[tuple[np.ndarray, int] | None],
    video_duration: float,
    ambient_path: Path | None,
) -> tuple[np.ndarray, int]:
    """Classic mode: time-fit each segment to fit in its SRT slot."""
    sr_master = next(r[1] for r in raw_tts if r is not None)
    fitted: list[tuple[np.ndarray, int] | None] = []
    n_capped = 0
    for i, (sub, r) in enumerate(zip(subs, raw_tts)):
        if r is None:
            fitted.append(None)
            continue
        audio, sr_seg = r
        target = (sub.end - sub.start).total_seconds()
        audio, capped = time_fit(audio, sr_seg, target)
        n_capped += int(capped)
        fitted.append((audio, sr_seg))
    if n_capped:
        print(f"  Time-fit capped at {MAX_SPEED_RATIO}x on {n_capped} segments")

    total_samples = int(video_duration * sr_master) + sr_master
    master = np.zeros(total_samples, dtype=np.float32)
    for sub, fit in zip(subs, fitted):
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
        master = master + AMBIENT_GAIN * amb.astype(np.float32)

    return master, sr_master


def _build_plan_dynamic(
    subs: list,
    raw_tts: list[tuple[np.ndarray, int] | None],
    video_duration: float,
    sr_master: int,
) -> list[dict]:
    """Build a list of timeline parts (gaps + segments) with the new durations."""
    import pyrubberband as pyrb
    parts: list[dict] = []
    cursor_orig = 0.0
    cursor_new = 0.0
    for i, sub in enumerate(subs):
        s_orig = sub.start.total_seconds()
        e_orig = sub.end.total_seconds()
        # Gap before this segment
        if cursor_orig < s_orig - 1e-3:
            gap_dur = s_orig - cursor_orig
            parts.append({
                "type": "gap",
                "orig_start": cursor_orig, "orig_end": s_orig,
                "new_start": cursor_new, "new_dur": gap_dur,
                "pts": 1.0, "audio": None,
            })
            cursor_new += gap_dur
        orig_slot = e_orig - s_orig
        r = raw_tts[i]
        if r is None or orig_slot <= 0:
            parts.append({
                "type": "seg", "tts_idx": i,
                "orig_start": s_orig, "orig_end": e_orig,
                "new_start": cursor_new, "new_dur": orig_slot,
                "pts": 1.0, "audio": None,
            })
            cursor_new += orig_slot
            cursor_orig = e_orig
            continue

        tts_audio, sr_seg = r
        tts_dur = len(tts_audio) / sr_seg
        desired_pts = tts_dur / orig_slot if orig_slot > 0 else 1.0

        if abs(desired_pts - 1.0) < 0.02:
            # Already matches within 2 %
            new_dur = orig_slot
            pts = 1.0
            final_audio = tts_audio
        elif desired_pts > 1.0:
            # TTS longer than slot -> slow the video down
            if desired_pts <= MAX_VIDEO_STRETCH:
                pts = desired_pts
                new_dur = orig_slot * pts
                final_audio = tts_audio
            else:
                # Cap reached: still need to compress audio a bit (last resort).
                pts = MAX_VIDEO_STRETCH
                new_dur = orig_slot * pts
                audio_speed = min(tts_dur / new_dur, MAX_AUDIO_COMPRESS)
                if audio_speed > 1.02:
                    final_audio = pyrb.time_stretch(tts_audio, sr_seg, audio_speed)
                else:
                    final_audio = tts_audio
        else:
            # TTS shorter than slot -> speed the video up. Audio stays natural.
            if desired_pts >= MIN_VIDEO_PTS:
                pts = desired_pts
                new_dur = orig_slot * pts
                final_audio = tts_audio
            else:
                # Cap reached: video at max speed-up, accept silence at the end
                # so we never have to mangle the audio for being too short.
                pts = MIN_VIDEO_PTS
                new_dur = orig_slot * pts
                final_audio = tts_audio

        parts.append({
            "type": "seg", "tts_idx": i,
            "orig_start": s_orig, "orig_end": e_orig,
            "new_start": cursor_new, "new_dur": new_dur,
            "pts": pts, "audio": final_audio,
        })
        cursor_new += new_dur
        cursor_orig = e_orig

    # Final tail
    if cursor_orig < video_duration - 1e-3:
        tail = video_duration - cursor_orig
        parts.append({
            "type": "gap",
            "orig_start": cursor_orig, "orig_end": video_duration,
            "new_start": cursor_new, "new_dur": tail,
            "pts": 1.0, "audio": None,
        })
    return parts


def _has_nvenc() -> bool:
    """Detect if ffmpeg's NVIDIA HEVC encoder is available (much faster on T4)."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        return "hevc_nvenc" in r.stdout
    except Exception:
        return False


def _warp_video(parts: list[dict], video_path: Path, work_dir: Path) -> Path:
    """Cut + setpts + concat the video according to the plan. Returns the warped video path."""
    part_dir = work_dir / "video_parts"
    part_dir.mkdir(exist_ok=True)
    part_files = []
    n_stretched = 0
    encoder = ("hevc_nvenc", "p5") if _has_nvenc() else ("libx265", "medium")
    enc_name, enc_preset = encoder
    src_fps = get_video_fps(video_path)
    print(f"  Warping {len(parts)} video parts (encoder: {enc_name}, {src_fps:.3f} fps)...", flush=True)
    for p in tqdm(parts, desc="  warp", leave=False, unit="part"):
        if p["orig_end"] - p["orig_start"] < 1e-4:
            continue
        j = len(part_files)
        out = part_dir / f"part_{j:04d}.mp4"
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{p['orig_start']:.3f}", "-to", f"{p['orig_end']:.3f}",
            "-i", str(video_path), "-an",
        ]
        if abs(p["pts"] - 1.0) > 1e-3:
            cmd += ["-filter:v", f"setpts={p['pts']:.4f}*PTS"]
            n_stretched += 1
        cmd += [
            "-c:v", enc_name, "-preset", enc_preset,
            "-pix_fmt", "yuv420p", "-r", f"{src_fps:.3f}",
            "-tag:v", "hvc1",
        ]
        if enc_name == "hevc_nvenc":
            cmd += ["-rc", "vbr", "-cq", "28", "-b:v", "0"]
        else:
            cmd += ["-crf", "28", "-x265-params", "log-level=error"]
        cmd += [str(out)]
        subprocess.run(cmd, check=True)
        part_files.append(out)

    print(f"  Video stretched on {n_stretched}/{len(parts)} parts")

    concat_list = work_dir / "concat.txt"
    with open(concat_list, "w") as f:
        for pf in part_files:
            f.write(f"file '{pf}'\n")
    warped = work_dir / "video_warped.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-c:v", "copy", str(warped),
        ],
        check=True,
    )
    return warped


def _build_master_dynamic(
    parts: list[dict],
    sr_master: int,
    ambient_path: Path | None,
) -> tuple[np.ndarray, int]:
    """Build audio on the NEW (warped) timeline."""
    import pyrubberband as pyrb
    total_dur = sum(p["new_dur"] for p in parts)
    master = np.zeros(int(total_dur * sr_master) + sr_master, dtype=np.float32)

    # Place TTS audios at their new positions
    for p in parts:
        if p["audio"] is None:
            continue
        start = int(p["new_start"] * sr_master)
        end = min(start + len(p["audio"]), len(master))
        master[start:end] += p["audio"][:end - start].astype(np.float32)

    # Warp ambient piecewise to match new timeline
    if ambient_path is not None:
        amb, sr_amb = sf.read(ambient_path)
        if amb.ndim > 1:
            amb = amb.mean(axis=1)
        if sr_amb != sr_master:
            from scipy import signal
            amb = signal.resample_poly(amb, sr_master, sr_amb)
        warped_amb = np.zeros_like(master)
        for p in parts:
            orig_a = int(p["orig_start"] * sr_master)
            orig_b = int(p["orig_end"] * sr_master)
            chunk = amb[orig_a:orig_b]
            if len(chunk) < 100:
                continue
            if abs(p["pts"] - 1.0) > 1e-3:
                try:
                    chunk = pyrb.time_stretch(chunk.astype(np.float32), sr_master, 1.0 / p["pts"])
                except Exception as e:
                    print(f"  [WARN] ambient stretch failed: {e}")
            new_a = int(p["new_start"] * sr_master)
            new_b = min(new_a + len(chunk), len(warped_amb))
            warped_amb[new_a:new_b] += chunk[:new_b - new_a].astype(np.float32)
        master = master + AMBIENT_GAIN * warped_amb

    return master, sr_master


def dub_one(
    item: VideoItem,
    tts: TTS,
    lang: str,
    work_dir: Path,
    output_path: Path,
    remove_voice: bool,
    dynamic_duration: bool = True,
    hq_mode: bool = False,
    custom_voice_ref: Path | None = None,
    cache_root: Path | None = None,
    skip_silent_segments: bool = True,
    burn_in_subs: bool = False,
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    seg_dir = work_dir / "segments"
    seg_dir.mkdir(exist_ok=True)

    # 1. Parse SRT
    subs = list(srt.parse(item.srt_path.read_text(encoding="utf-8")))
    mode_tags = ["dynamic-duration" if dynamic_duration else "classic"]
    if hq_mode:
        mode_tags.append("HQ")
    if custom_voice_ref:
        mode_tags.append("custom-ref")
    print(f"  SRT: {len(subs)} segments  |  mode: {' + '.join(mode_tags)}")

    # 2-3. Video + orig audio + optional ambient/vocals (with cross-language cache)
    video_path, orig_audio, ambient_path, vocals_path = prepare_video_and_ambient(
        item.url, work_dir, cache_root, remove_voice
    )

    # 4. Voice reference. Priority:
    #    1) User-uploaded clean sample
    #    2) Demucs-isolated vocals (clean voice without ambient bleed)
    #    3) Raw audio extracted from the longest SRT segment (legacy fallback)
    ref_path = work_dir / "ref.wav"
    if custom_voice_ref is not None and Path(custom_voice_ref).exists():
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(custom_voice_ref),
                "-ac", "1", "-ar", str(SAMPLE_RATE), str(ref_path),
            ],
            check=True,
        )
        amb_meta = sf.info(ref_path)
        print(f"  Voice ref (custom): {custom_voice_ref.name} ({amb_meta.duration:.1f}s)")
    elif vocals_path is not None and vocals_path.exists():
        ref_s, ref_e = extract_voice_ref(vocals_path, subs, ref_path)
        print(f"  Voice ref (Demucs-cleaned): {ref_s:.1f}-{ref_e:.1f}s")
    else:
        ref_s, ref_e = extract_voice_ref(orig_audio, subs, ref_path)
        print(f"  Voice ref (from video): {ref_s:.1f}-{ref_e:.1f}s")

    # 5b. Optionally skip segments where the original speaker was silent
    voice_mask: list[bool] | None = None
    if skip_silent_segments:
        voice_mask = detect_silent_segments(orig_audio, subs)
        n_skipped = sum(1 for m in voice_mask if not m)
        if n_skipped:
            print(f"  Skipping TTS for {n_skipped}/{len(subs)} segments where the original is silent")

    # 6. Generate TTS at natural speed
    raw_tts = _generate_tts_raw(subs, tts, lang, ref_path, seg_dir, hq_mode=hq_mode, voice_mask=voice_mask)
    if not any(r is not None for r in raw_tts):
        raise RuntimeError("No TTS generated for any segment")
    sr_master = next(r[1] for r in raw_tts if r is not None)

    # 7. Build video + master audio (branch on mode)
    video_duration = get_video_duration(video_path)
    if dynamic_duration:
        parts = _build_plan_dynamic(subs, raw_tts, video_duration, sr_master)
        final_video = _warp_video(parts, video_path, work_dir)
        master, _ = _build_master_dynamic(parts, sr_master, ambient_path)
    else:
        final_video = video_path
        master, _ = _build_master_classic(subs, raw_tts, video_duration, ambient_path)

    # 8. Normalize
    peak = float(np.max(np.abs(master)))
    if peak > 0.99:
        master *= 0.99 / peak
    master_path = work_dir / "master.wav"
    sf.write(master_path, master, sr_master)

    # 9. Mux
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if burn_in_subs:
        srt_arg = str(item.srt_path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        mux_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(final_video),
            "-i", str(master_path),
            "-filter_complex", f"[0:v]subtitles='{srt_arg}'[v]",
            "-map", "[v]", "-map", "1:a:0",
            "-c:a", "aac", "-b:a", "192k",
            "-c:v", "hevc_nvenc" if _has_nvenc() else "libx265",
        ]
        if _has_nvenc():
            mux_cmd += ["-preset", "p5", "-rc", "vbr", "-cq", "28", "-b:v", "0", "-tag:v", "hvc1"]
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
    dynamic_duration: bool = True,
    hq_mode: bool = False,
    custom_voice_ref: str | Path | None = None,
    cache_root: str | Path | None = "/tmp/sta-ru-cache",
    skip_silent_segments: bool = True,
    burn_in_subs: bool = False,
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

    # Load TTS once (heavy deps imported lazily so batch_dub_edge doesn't need them)
    import torch
    from TTS.api import TTS
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
                dynamic_duration=dynamic_duration,
                hq_mode=hq_mode,
                custom_voice_ref=Path(custom_voice_ref) if custom_voice_ref else None,
                cache_root=Path(cache_root) if cache_root else None,
                skip_silent_segments=skip_silent_segments,
                burn_in_subs=burn_in_subs,
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
    ap.add_argument("--no-dynamic-duration", action="store_true",
                    help="Disable dynamic duration (stretches video to fit dubbed audio); falls back to classic time-fit on audio")
    ap.add_argument("--hq", action="store_true",
                    help="High-quality TTS mode: beam search + low temperature (~3x slower, better dubbing)")
    ap.add_argument("--voice-ref", type=str, default=None,
                    help="Optional .wav with a clean voice sample to clone from (replaces auto-extraction from video)")
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
        dynamic_duration=not args.no_dynamic_duration,
        hq_mode=args.hq,
        custom_voice_ref=args.voice_ref,
    )


if __name__ == "__main__":
    _cli()
