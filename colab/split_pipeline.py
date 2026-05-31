"""
Sta-RU split pipeline: cut a long video at silences and dub each part
independently, then concatenate the dubbed parts.

Why this exists: Demucs, edge-tts and the video warp all scale with video
duration. On a stock 12 GB Colab they each get fragile past ~30 minutes — most
visibly Demucs, which gets OOM-killed (sometimes taking the kernel with it).
The fix is to split *before* the pipeline: each sub-part goes through the
normal dub_one path as if it were a short standalone video, and we glue the
dubbed outputs together at the end.

This module is a thin adapter over `video_pipeline.py` (mirror of the splitter
in the Lazy-Money/A repo). Reuses its silence detection (ffmpeg silencedetect
+ silero-VAD) and its concat helpers. Anything specific to our dubbing
pipeline — cut planning without speed-up, SRT slicing, end-to-end glue — lives
here, so video_pipeline.py stays a drop-in copy from the upstream repo.
"""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import video_pipeline as vp


# Max sub-part duration (seconds). 35 minutes — enough margin under Demucs's
# OOM threshold on stock 12 GB Colab, while keeping splits to 2 parts for
# videos up to 70 min (which is most of the catalog).
DEFAULT_MAX_PART_SECONDS = 35 * 60


def _trust_silero_vad_repo() -> None:
    """torch.hub.load prompts "Do you trust this repository? (y/N)" the first
    time it loads a third-party repo. In a Colab cell stdin isn't connected
    to the user, so the prompt hangs the kernel forever — `_load_vad` in
    video_pipeline can't catch it because nothing raises. We pre-write the
    repo into torch's trusted_list so the prompt is skipped entirely.
    """
    try:
        import torch.hub
        hub_dir = Path(torch.hub.get_dir())
    except Exception:
        return  # no torch -> VAD won't run anyway, ffmpeg-only is fine
    trusted_file = hub_dir / "trusted_list"
    entry = "snakers4_silero-vad"
    try:
        trusted_file.parent.mkdir(parents=True, exist_ok=True)
        existing = trusted_file.read_text().splitlines() if trusted_file.exists() else []
        if entry not in existing:
            with trusted_file.open("a") as f:
                f.write(entry + "\n")
    except OSError:
        pass


def needs_split(video_path: Path, max_part_seconds: int = DEFAULT_MAX_PART_SECONDS) -> bool:
    """True if the video is long enough to warrant pre-splitting."""
    dur = vp.get_duration(video_path) or 0.0
    return dur > max_part_seconds


def _srt_intervals(srt_path: Path) -> list[tuple[float, float]]:
    """Load SRT and return [(start_s, end_s), ...] of every sub, sorted."""
    import srt as srt_lib
    subs = list(srt_lib.parse(srt_path.read_text(encoding="utf-8")))
    return sorted(
        (s.start.total_seconds(), s.end.total_seconds()) for s in subs
    )


def _pause_in_srt_gap(pause: dict, srt_intervals: list[tuple[float, float]]) -> bool:
    """True iff `pause` (a dict with 'start'/'end') overlaps no SRT sub. Used
    to prefer cuts where no caption is active, so split_srt doesn't have to
    decide which side a straddling sub belongs to."""
    p_s, p_e = pause["start"], pause["end"]
    for s, e in srt_intervals:
        if e <= p_s:
            continue   # sub ended before pause starts
        if s >= p_e:
            break      # sub starts after pause ends; rest is too far right
        return False   # any overlap disqualifies
    return True


def plan_cuts(video_path: Path, max_part_seconds: int = DEFAULT_MAX_PART_SECONDS,
              srt_path: Path | None = None) -> list[float]:
    """Pick cut timestamps (in seconds) so every resulting part is <=
    max_part_seconds. Cuts land at the midpoint of detected silences whenever
    possible. Returns an empty list when the video already fits in one part.

    When `srt_path` is provided, only silences that fall in an SRT gap (no
    active caption) are considered — that way no subtitle straddles a cut and
    we never have to duplicate or split a line across two parts.

    Speed-up is intentionally *not* applied here — our SRT must stay aligned
    to the video frame-perfectly, and atempo'ing the audio against an
    untouched SRT would desync.
    """
    duration = vp.get_duration(video_path) or 0.0
    if duration <= max_part_seconds:
        return []

    # Detect silences with both backends, then merge (the user's video_pipeline
    # already does this; we just call its primitives). Trust the silero-vad
    # repo first so torch.hub doesn't hang on the y/N prompt in Colab.
    _trust_silero_vad_repo()
    ff_pauses = vp.detect_silences_ffmpeg(video_path)
    vad_pauses = vp.detect_silences_vad(video_path)
    pauses = vp.merge_pauses(ff_pauses, vad_pauses)

    srt_intervals = _srt_intervals(srt_path) if srt_path else []
    pauses_in_gap = (
        [p for p in pauses if _pause_in_srt_gap(p, srt_intervals)]
        if srt_intervals else pauses
    )

    n_parts = max(2, math.ceil(duration / max_part_seconds))
    cuts: list[float] = []
    prev_cut = 0.0

    for i in range(1, n_parts):
        # Acceptable cut window for this part: enough room left for what
        # remains, and not so far that this part exceeds max_part_seconds.
        remaining = n_parts - i
        min_cut = max(prev_cut + 1.0, duration - remaining * max_part_seconds)
        max_cut = prev_cut + max_part_seconds

        # First pass: gap-aware silences. Second pass (fallback): any
        # silence. Last resort: hard cut in the middle of the window.
        best_cut: float | None = None
        for candidates in (pauses_in_gap, pauses):
            best_pause_dur = -1.0
            for p in candidates:
                mid = (p["start"] + p["end"]) / 2.0
                if mid < min_cut or mid > max_cut:
                    continue
                if p["duration"] > best_pause_dur:
                    best_pause_dur = p["duration"]
                    best_cut = mid
            if best_cut is not None:
                break
        if best_cut is None:
            best_cut = (min_cut + max_cut) / 2.0
        cuts.append(round(best_cut, 3))
        prev_cut = best_cut

    return cuts


def split_video(video_path: Path, cuts: list[float], out_dir: Path) -> tuple[list[Path], list[float]]:
    """Cut video_path into len(cuts)+1 parts using ffmpeg's segment muxer.
    Returns (part_paths_in_order, actual_cuts).

    Why the segment muxer (and not `-ss/-to ... -c copy` per part):

      With per-part `-ss/-to`, parts 2..N inherit the source video's pts —
      part 2 of a 51-min video starts at pts=1182s, not pts=0. Downstream
      tools (Demucs reads "audio starts 1182s in"; the ffmpeg mux at the
      end of dub_one stretches the TTS track to that pts range) then
      misinterpret the timing and the concatenated output ends up with
      audio that plays faster than the video.

      The segment muxer with `-reset_timestamps 1` forces every part to
      start at pts=0, which is what every downstream tool expects.

    The trade-off: segment cuts always land on a keyframe at or *after*
    the requested time, so part boundaries may shift by up to a GOP (~2-4s
    for typical YouTube encodes). We return the *actual* cuts (measured
    from probed part durations) so split_srt can align captions to the
    real boundaries.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / f"{video_path.stem}_part_%03d{video_path.suffix}")
    times_csv = ",".join(f"{c:.3f}" for c in cuts)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-map", "0",
        "-c", "copy",
        "-f", "segment",
        "-segment_times", times_csv,
        "-segment_start_number", "1",
        "-reset_timestamps", "1",
        "-avoid_negative_ts", "make_zero",
        pattern,
    ]
    subprocess.run(cmd, check=True)

    parts = sorted(out_dir.glob(f"{video_path.stem}_part_*{video_path.suffix}"))
    if not parts:
        raise RuntimeError(f"segment muxer produced no parts in {out_dir}")
    if len(parts) != len(cuts) + 1:
        raise RuntimeError(
            f"expected {len(cuts) + 1} parts, got {len(parts)} in {out_dir}"
        )

    # Probe actual durations so we know where the keyframe-aligned cuts
    # really fell. split_srt needs these (not the requested cuts) to keep
    # captions in sync.
    actual_cuts: list[float] = []
    cum = 0.0
    for p in parts[:-1]:
        d = vp.get_duration(p) or 0.0
        cum += d
        actual_cuts.append(round(cum, 3))
    return parts, actual_cuts


def split_srt(srt_path: Path, cuts: list[float], out_dir: Path,
              video_duration: float | None = None) -> list[Path]:
    """Slice srt_path at the same cut timestamps. Each output .srt contains
    only the entries that fall in its part, with timestamps shifted so the
    part starts at 00:00. Entries that straddle a cut are clamped to fit.
    Returns the list of part srt paths in order.
    """
    import srt as srt_lib  # already a dep of the dubbing pipeline
    from datetime import timedelta

    out_dir.mkdir(parents=True, exist_ok=True)
    text = srt_path.read_text(encoding="utf-8")
    all_subs = list(srt_lib.parse(text))

    boundaries_s = [0.0] + list(cuts) + [video_duration or float("inf")]
    n_parts = len(boundaries_s) - 1
    # Bucket each sub into exactly one part by its midpoint. plan_cuts (with
    # srt_path) already tries to put cuts in SRT gaps so this assignment is
    # usually clean; when a sub still straddles a cut, the side containing
    # the *middle* of the sub wins. Clamp the resulting timestamps into the
    # part's [0, dur] window so the dub pipeline sees valid timing.
    buckets: list[list] = [[] for _ in range(n_parts)]
    for sub in all_subs:
        s = sub.start.total_seconds()
        e = sub.end.total_seconds()
        mid = (s + e) / 2.0
        idx = 0
        for i in range(n_parts):
            if boundaries_s[i] <= mid < boundaries_s[i + 1]:
                idx = i
                break
        else:
            idx = n_parts - 1  # mid past last boundary — last part
        part_start = boundaries_s[idx]
        part_end = boundaries_s[idx + 1]
        s_clamped = max(s, part_start)
        e_clamped = min(e, part_end)
        if e_clamped <= s_clamped:
            continue
        buckets[idx].append(srt_lib.Subtitle(
            index=len(buckets[idx]) + 1,
            start=timedelta(seconds=s_clamped - part_start),
            end=timedelta(seconds=e_clamped - part_start),
            content=sub.content,
        ))

    out_paths: list[Path] = []
    for i, part_subs in enumerate(buckets):
        part_srt = out_dir / f"{srt_path.stem}_part_{i + 1:03d}.srt"
        part_srt.write_text(srt_lib.compose(part_subs), encoding="utf-8")
        out_paths.append(part_srt)
    return out_paths


def concat_dubbed_parts(part_paths: list[Path], out_path: Path) -> bool:
    """Concatenate dubbed video parts into a single output. Tries stream-copy
    (concat demuxer) first; falls back to the concat filter (re-encode) if
    streams don't align. Returns True on success.
    """
    if not part_paths:
        return False
    if len(part_paths) == 1:
        shutil.copy(part_paths[0], out_path)
        return True

    if vp.streams_compatible(part_paths):
        if vp.concat_streamcopy(part_paths, out_path):
            return True
        # If stream-copy reports failure (rare with compatible streams), fall
        # through to the filter path rather than giving up.

    # Fallback: concat filter with re-encode. We pass settings/probe shaped
    # for vp.concat_filter (compress=False keeps it on libx264).
    fake_settings = {"compress": False, "quality_crf": 22}
    fake_probe = {"path": part_paths[0], "width": None, "height": None,
                  "duration": None, "bit_rate": None, "ok": True}
    return vp.concat_filter(part_paths, out_path, fake_settings, fake_probe)
