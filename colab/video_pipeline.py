#!/usr/bin/env python3
"""
VideoPipeline - unified video processor for dubbing workflow.

Phases implemented so far:
  1. Interactive skeleton (prompts, validation, dependency check, preview).
  2. Video discovery + probing (duration, resolution) and output path policy.

Real video processing lands in Phase 3+.

Required binaries in PATH: ffmpeg, ffprobe.
Optional Python packages: torch, torchaudio (enables VAD double-check).
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Terminal styling
# ---------------------------------------------------------------------------

if os.name == "nt":
    os.system("")  # enable ANSI escape sequences on modern Windows terminals


class C:
    R = "\033[0m"
    B = "\033[1m"
    DIM = "\033[2m"
    G = "\033[32m"
    RED = "\033[31m"
    Y = "\033[33m"
    CY = "\033[36m"
    # Truecolor orange for question text (approx. Claude's brand warm orange).
    ORANGE = "\033[38;2;232;145;91m"


# OSC 11 sets the terminal background; OSC 111 resets it. Modern terminals
# (Windows Terminal, iTerm2, Kitty, Alacritty, ...) honour these; legacy
# conhost/cmd simply ignores them, so they are safe to emit unconditionally.
BG_DARK = "\033]11;#1a1a1a\033\\"
BG_RESET = "\033]111\033\\"


def set_dark_background() -> None:
    sys.stdout.write(BG_DARK)
    sys.stdout.flush()


def reset_background() -> None:
    sys.stdout.write(BG_RESET)
    sys.stdout.flush()


def disable_quick_edit_mode() -> None:
    """Turn off Windows cmd's 'Quick Edit Mode' so clicking or selecting
    text in the console no longer freezes the output stream while ffmpeg
    is encoding. Without this, a single click is enough to make users
    believe the program is hung. No-op on non-Windows platforms.

    Side effect: classic drag-to-select stops working in cmd.exe; Windows
    Terminal users can still copy with Ctrl+Shift+C."""
    if os.name != "nt":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        STD_INPUT_HANDLE = -10
        ENABLE_QUICK_EDIT = 0x0040
        ENABLE_EXTENDED_FLAGS = 0x0080
        handle = kernel32.GetStdHandle(STD_INPUT_HANDLE)
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            new_mode = (mode.value | ENABLE_EXTENDED_FLAGS) & ~ENABLE_QUICK_EDIT
            kernel32.SetConsoleMode(handle, new_mode)
    except Exception:
        pass  # not critical; just a UX improvement


def confirm_exit_or_skip() -> bool:
    """Called when KeyboardInterrupt is caught mid-batch. Returns True if
    the user wants to exit, False to skip the current item and continue.
    A second Ctrl+C while answering forces an immediate exit."""
    print()
    try:
        answer = input(
            f"  {C.Y}Do you want to Skip or Exit?{C.R} "
            f"{C.ORANGE}[S/E] (default: S){C.R} "
        ).strip().lower()
    except (KeyboardInterrupt, EOFError):
        return True
    return answer in ("e", "exit")


def ok(msg: str) -> None:
    print(f"  {C.G}[OK]{C.R} {msg}")


def bad(msg: str) -> None:
    print(f"  {C.RED}[!!]{C.R} {msg}")


def warn(msg: str) -> None:
    print(f"  {C.Y}[..]{C.R} {msg}")


def info(msg: str) -> None:
    print(f"  {C.CY}[>>]{C.R} {msg}")


def head(title: str) -> None:
    print(f"\n{C.B}{C.CY}-- {title} --{C.R}\n")


BANNER = f"""
{C.B}{C.CY}============================================================
        Hey, let's gonna process those videos!
============================================================{C.R}
"""


# ---------------------------------------------------------------------------
# Input helpers (all validated, all re-prompt on invalid input)
# ---------------------------------------------------------------------------

def _read(prompt: str) -> str:
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(130)


def _q(text: str) -> str:
    """Paint a question in orange; reset afterwards so user input is default color."""
    return f"{C.ORANGE}{text}{C.R}"


def ask_yes_no(question: str, default: Optional[str] = None) -> bool:
    hint = "[Y/N]"
    if default:
        hint = f"[Y/N] (default: {default.upper()})"
    print()
    while True:
        raw = _read(f"  {_q(question)} {_q(hint)} ").strip().lower()
        if not raw and default:
            return default.upper() == "Y"
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        warn("Please answer Y or N.")


def ask_choice(question: str, options: dict, default: Optional[str] = None) -> str:
    keys = list(options.keys())
    hint = "[" + "/".join(keys) + "]"
    if default:
        hint = f"{hint} (default: {default})"
    print()
    print(f"  {_q(question)}")
    for k, v in options.items():
        marker = " (default)" if default and k == default else ""
        print(f"    [{k}] {v}{marker}")
    while True:
        raw = _read(f"  {_q('Choice')} {_q(hint)}: ").strip()
        if not raw and default:
            return default
        for k in keys:
            if raw.lower() == k.lower():
                return k
        warn("Please answer one of: " + ", ".join(keys))


def ask_int_range(question: str, lo: int, hi: int, default: Optional[int] = None) -> int:
    hint = f"[{lo}-{hi}]"
    if default is not None:
        hint += f" (default {default})"
    print()
    while True:
        raw = _read(f"  {_q(question)} {_q(hint)}: ").strip()
        if not raw and default is not None:
            return default
        try:
            n = int(raw)
        except ValueError:
            warn("Not a valid integer.")
            continue
        if lo <= n <= hi:
            return n
        warn(f"Out of range. Must be between {lo} and {hi}.")


def ask_path(question: str, default: Optional[Path] = None) -> Path:
    print()
    while True:
        print(f"  {_q(question)}")
        if default is not None:
            print(f"  {C.DIM}(press Enter to use: {default}){C.R}")
        raw = _read(f"  {_q('Path:')} ").strip().strip('"').strip("'")
        if not raw and default is not None:
            return default.resolve()
        if not raw:
            warn("Path cannot be empty.")
            continue
        p = Path(raw).expanduser()
        if not p.exists():
            bad(f"Path does not exist: {p}")
            continue
        if not p.is_dir():
            bad(f"Path is not a folder: {p}")
            continue
        return p.resolve()


def ask_str(question: str, validator=None) -> str:
    print()
    while True:
        raw = _read(f"  {_q(question)}: ").strip()
        if not raw:
            warn("Cannot be empty.")
            continue
        if validator:
            err = validator(raw)
            if err:
                warn(err)
                continue
        return raw


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def _has_binary(name: str) -> bool:
    return shutil.which(name) is not None


def _has_module(name: str) -> bool:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            __import__(name)
            return True
        except ImportError:
            return False


def _pip_install(*pkgs: str, extra_args: Optional[list] = None) -> bool:
    cmd = [sys.executable, "-m", "pip", "install", *pkgs]
    if extra_args:
        cmd += extra_args
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def check_dependencies() -> None:
    head("Checking dependencies")

    missing_bin = [b for b in ("ffmpeg", "ffprobe") if not _has_binary(b)]
    for b in ("ffmpeg", "ffprobe"):
        if b in missing_bin:
            bad(f"{b} NOT found in PATH")
        else:
            ok(f"{b} found")

    if missing_bin:
        print()
        bad("Required binaries are missing.")
        info("Install ffmpeg (which ships with ffprobe) from https://ffmpeg.org/download.html")
        info("Then make sure both are reachable from your PATH.")
        sys.exit(1)

    if _has_module("torch") and _has_module("torchaudio"):
        ok("torch + torchaudio found (VAD double-check available)")
        if not _has_module("numpy"):
            warn("numpy not installed")
            print(f"  {C.DIM}  (numpy is a numerical-math library that torch uses internally."
                  f"{C.R}")
            print(f"  {C.DIM}   This pipeline works fine without it - the only effect is a"
                  f"{C.R}")
            print(f"  {C.DIM}   cosmetic 'Failed to initialize NumPy' warning on startup.){C.R}")
            if ask_yes_no("Install numpy now to silence that warning?", default="Y"):
                if _pip_install("numpy"):
                    ok("numpy installed")
                else:
                    warn("Installation failed - the warning will keep showing but VAD still works")
    else:
        warn("torch / torchaudio not installed - silence detection will use ffmpeg only")
        if ask_yes_no("    Install torch + torchaudio now (CPU build)?", default="N"):
            numpy_ok = _has_module("numpy") or _pip_install("numpy")
            if not numpy_ok:
                warn("Could not install numpy - torch will still work but with a warning.")
            if _pip_install(
                "torch", "torchaudio",
                extra_args=["--index-url", "https://download.pytorch.org/whl/cpu"],
            ):
                ok("torch + torchaudio installed")
            else:
                warn("Installation failed - continuing without VAD")

    # Optional: chardet (or charset_normalizer) for subtitle encoding
    # detection. Without it the pipeline falls back to a manual UTF-8 -> ...
    # -> Latin-1 chain that misreads some legacy CP1251 (Cyrillic) files.
    if _has_module("chardet") or _has_module("charset_normalizer"):
        name = "chardet" if _has_module("chardet") else "charset_normalizer"
        ok(f"{name} found (auto-detect subtitle encoding)")
    else:
        warn("chardet not installed")
        print(f"  {C.DIM}  Helps decode subtitles in legacy encodings"
              f" (CP1251 / CP1252 / GB18030 / ...){C.R}")
        print(f"  {C.DIM}  when UTF-8 fails. Without it modern UTF-8 subs still"
              f" work fine.{C.R}")
        if ask_yes_no("Install chardet now?", default="Y"):
            if _pip_install("chardet"):
                ok("chardet installed")
            else:
                warn("Installation failed - using fallback encoding chain")

    # Encoder autodetection - runs last so GPU probes do not slow down the
    # other checks, and we can prompt the user right after showing what we
    # found.  GPU only affects the H.265 compression path now - resize /
    # scale / speed re-encodes always use CPU libx264 because it is more
    # efficient per bit and fast enough for those short operations.
    print()
    info("Detecting available encoders (GPU probe may take a few seconds)...")
    detect_encoders()
    gpu_h265_available = DETECTED["h265_encoder"] != "libx265"
    if gpu_h265_available:
        ok("This program uses the H.265 codec. You can use the CPU")
        print("       (which is slow and reduces file size to about 66% of")
        print("       the original - not all cases) or you can use the GPU")
        print("       (which is around 5 times faster, but reduces file size")
        print("       to only about 20% of the original).")
        DETECTED["gpu_enabled"] = ask_yes_no(
            "Use the GPU for compression (H.265)?", default="N"
        )
        if DETECTED["gpu_enabled"]:
            info("GPU enabled - compression will use " + DETECTED['h265_encoder'])
        else:
            info("Using CPU libx265")
    else:
        info("No GPU H.265 encoder available - compression will use CPU libx265")


# ---------------------------------------------------------------------------
# Settings collection
# ---------------------------------------------------------------------------

H265_CRF_MIN = 18
H265_CRF_MAX = 32
H265_CRF_DEFAULT = 22

# Unified output quality range - applies to H.264 and H.265 re-encodes.
QUALITY_CRF_MIN = 18
QUALITY_CRF_MAX = 32
QUALITY_CRF_DEFAULT = 22

CONTAINER_CHOICES = {"1": ".mp4", "2": ".mov", "3": ".mkv"}
SCALE_CHOICES = {"1": "480p", "2": "720p", "3": "1080p"}


def _validate_suffix(s: str) -> Optional[str]:
    bad_chars = set('\\/:*?"<>|')
    if any(c in bad_chars for c in s):
        return 'Invalid characters. Avoid: \\ / : * ? " < > |'
    return None


def ask_output_policy(default_suffix: str = "_mod") -> dict:
    if ask_yes_no("Should we overwrite old videos?", default="N"):
        return {"overwrite": True, "suffix": ""}
    change = ask_yes_no(
        f'New videos will be named with "{default_suffix}" at the end. '
        f'Do you want to change it?',
        default="N",
    )
    if not change:
        return {"overwrite": False, "suffix": default_suffix}
    suffix = ask_str(
        "How should we call it? (don't include the extension, e.g. dubbed)",
        validator=_validate_suffix,
    )
    if not suffix.startswith("_"):
        suffix = "_" + suffix
    return {"overwrite": False, "suffix": suffix}


def ask_compress_enable(default: str = "Y") -> bool:
    print(f"  {C.DIM}Re-encodes with H.265 only on videos whose source bitrate is{C.R}")
    print(f"  {C.DIM}above the threshold for their resolution{C.R}")
    print(f"  {C.DIM}  (e.g. >= 1.5 Mbps for 720p, >= 3 Mbps for 1080p).{C.R}")
    print(f"  {C.DIM}Smaller / already-compressed videos pass through untouched.{C.R}")
    return ask_yes_no("Compress video bitrate?", default=default)


def ask_output_quality() -> int:
    print(f"  {C.DIM}Compression quality (CRF). Lower = less compression, bigger file.{C.R}")
    print(f"  {C.DIM}  20 = minimal compression (closer to original size){C.R}")
    print(f"  {C.DIM}  22 = high quality, balanced (recommended){C.R}")
    print(f"  {C.DIM}  26 = good, noticeably smaller{C.R}")
    print(f"  {C.DIM}  32 = aggressive, acceptable quality{C.R}")
    return ask_int_range(
        "Compression quality (CRF)",
        QUALITY_CRF_MIN,
        QUALITY_CRF_MAX,
        QUALITY_CRF_DEFAULT,
    )


def ask_container() -> Optional[str]:
    if not ask_yes_no("Change container format?", default="N"):
        return None
    key = ask_choice("Target container?", CONTAINER_CHOICES, default="1")
    return CONTAINER_CHOICES[key]


def _reask_resize_small(s: dict) -> dict:
    s["resize_small"] = ask_yes_no(
        "Check and resize videos under 640x480?", default="Y"
    )
    return s


def _reask_process_long(s: dict) -> dict:
    s["process_long"] = ask_yes_no(
        "Process all videos over 1799s (~30 min)?", default="Y"
    )
    return s


def _reask_scale_down(s: dict) -> dict:
    if ask_yes_no("Scale down big videos (4K/1080p/720p)?", default="N"):
        choice = ask_choice("Target resolution?", SCALE_CHOICES, default="2")
        s["scale_down"] = SCALE_CHOICES[choice]
    else:
        s["scale_down"] = None
    return s


def _reask_compress(s: dict) -> dict:
    # Different default per mode: To-Dub videos arrive raw and benefit from
    # compression, while Already-Dubbed videos come back from the platform
    # already processed and usually do not need a second pass.
    default = "N" if s.get("mode") == "already_dubbed" else "Y"
    s["compress"] = ask_compress_enable(default=default)
    if s["compress"] and s["quality_crf"] is None:
        s["quality_crf"] = ask_output_quality()
    elif not s["compress"]:
        s["quality_crf"] = None
    return s


def _reask_crf(s: dict) -> dict:
    s["quality_crf"] = ask_output_quality()
    return s


def _reask_container(s: dict) -> dict:
    s["container"] = ask_container()
    return s


def _reask_overwrite(s: dict) -> dict:
    s["policy"] = ask_output_policy()
    return s


def _reask_suffix(s: dict) -> dict:
    suffix = ask_str(
        "How should we call it? (don't include the extension, e.g. dubbed)",
        validator=_validate_suffix,
    )
    if not suffix.startswith("_"):
        suffix = "_" + suffix
    s["policy"]["suffix"] = suffix
    return s


def _reask_join(s: dict) -> dict:
    s["join"] = ask_yes_no(
        "Join split segments (files ending in _01, _02, ...)?", default="Y"
    )
    if not s["join"]:
        s["merge_subs"] = False
    return s


def _reask_merge_subs(s: dict) -> dict:
    s["merge_subs"] = ask_yes_no(
        "Merge subtitles? (.srt / .vtt next to each segment, same stem)",
        default="Y",
    )
    return s


def ask_output_dir() -> Optional[Path]:
    """Return None when outputs should sit next to their source video, or a
    Path for a custom destination folder. Creates the folder if it does
    not exist (the user explicitly typed it, so this is APB-friendly)."""
    same = ask_yes_no(
        "Save outputs in the same folder as the source videos?", default="Y"
    )
    if same:
        return None
    while True:
        raw = _read(f"  {_q('Output folder path:')} ").strip().strip('"').strip("'")
        if not raw:
            warn("Path cannot be empty.")
            continue
        p = Path(raw).expanduser()
        if p.exists():
            if not p.is_dir():
                bad(f"Path exists but is not a folder: {p}")
                continue
            return p.resolve()
        try:
            p.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            bad(f"Could not create folder: {e}")
            continue
        ok(f"Created folder: {p.resolve()}")
        return p.resolve()


def _reask_output_dir(s: dict) -> dict:
    s["output_dir"] = ask_output_dir()
    return s


def _to_dub_rows(s: dict) -> list:
    """Review rows for the To-Dub branch. Each row is (label, value, reask_fn).
    Rows whose underlying answer is irrelevant in the current state are
    simply not included (e.g. CRF when compress=N, Suffix when overwrite=Y)."""
    rows = [
        ("Resize under 640x480",
         "Y" if s["resize_small"] else "N", _reask_resize_small),
        ("Process over 1799s",
         "Y" if s["process_long"] else "N", _reask_process_long),
        ("Scale down",
         s["scale_down"] or "N", _reask_scale_down),
        ("Compress bitrate",
         "Y" if s["compress"] else "N", _reask_compress),
    ]
    if s["compress"]:
        rows.append(
            ("Compression quality (CRF)",
             str(s["quality_crf"]), _reask_crf)
        )
    rows.append(
        ("Change container", s["container"] or "N", _reask_container)
    )
    rows.append(
        ("Overwrite originals",
         "Y" if s["policy"]["overwrite"] else "N", _reask_overwrite)
    )
    if not s["policy"]["overwrite"]:
        rows.append(
            ("Suffix", s["policy"]["suffix"], _reask_suffix)
        )
        rows.append(
            ("Output folder",
             "same as source" if s.get("output_dir") is None else str(s["output_dir"]),
             _reask_output_dir)
        )
    return rows


def _already_dubbed_rows(s: dict) -> list:
    rows = [
        ("Join split segments",
         "Y" if s["join"] else "N", _reask_join),
    ]
    if s["join"]:
        rows.append(
            ("Merge subtitles",
             "Y" if s.get("merge_subs") else "N", _reask_merge_subs)
        )
    rows.append(
        ("Compress bitrate",
         "Y" if s["compress"] else "N", _reask_compress)
    )
    if s["compress"]:
        rows.append(
            ("Compression quality (CRF)",
             str(s["quality_crf"]), _reask_crf)
        )
    rows.append(
        ("Change container", s["container"] or "N", _reask_container)
    )
    rows.append(
        ("Overwrite originals",
         "Y" if s["policy"]["overwrite"] else "N", _reask_overwrite)
    )
    if not s["policy"]["overwrite"]:
        rows.append(
            ("Suffix", s["policy"]["suffix"], _reask_suffix)
        )
        rows.append(
            ("Output folder",
             "same as source" if s.get("output_dir") is None else str(s["output_dir"]),
             _reask_output_dir)
        )
    return rows


def review_and_edit(s: dict, rows_fn) -> dict:
    """Loop: show the answer table, let the user pick a row to re-ask, or
    press Enter to confirm and continue."""
    while True:
        head("Review your answers")
        rows = rows_fn(s)
        label_w = max(len(label) for label, _, _ in rows)
        for i, (label, val, _) in enumerate(rows, 1):
            print(f"  {i}. {label:<{label_w}}   {val}")
        print()
        raw = _read(
            f"  {_q('Enter a number to change, or Enter to continue')}: "
        ).strip()
        if not raw:
            return s
        try:
            n = int(raw)
        except ValueError:
            warn(f"Enter a number 1-{len(rows)} or just press Enter.")
            continue
        if n < 1 or n > len(rows):
            warn(f"Out of range. Valid: 1-{len(rows)}.")
            continue
        _, _, reask_fn = rows[n - 1]
        s = reask_fn(s)


def collect_to_dub_settings() -> dict:
    head("To Dub - configuration")
    s = {
        "mode": "to_dub",
        "resize_small": False,
        "scale_down": None,
        "process_long": False,
        "compress": False,
        "quality_crf": None,
        "container": None,
        "policy": None,
        "output_dir": None,
    }
    _reask_resize_small(s)
    _reask_process_long(s)
    _reask_scale_down(s)
    _reask_compress(s)
    _reask_container(s)
    s["policy"] = ask_output_policy()
    if not s["policy"]["overwrite"]:
        s["output_dir"] = ask_output_dir()
    return review_and_edit(s, _to_dub_rows)


def collect_already_dubbed_settings() -> dict:
    head("Already Dubbed - configuration")
    s = {
        "mode": "already_dubbed",
        "join": False,
        "merge_subs": False,
        "compress": False,
        "quality_crf": None,
        "container": None,
        "policy": None,
        "output_dir": None,
    }
    _reask_join(s)
    if s["join"]:
        _reask_merge_subs(s)
    _reask_compress(s)
    _reask_container(s)
    s["policy"] = ask_output_policy(default_suffix="_joined")
    if not s["policy"]["overwrite"]:
        s["output_dir"] = ask_output_dir()
    return review_and_edit(s, _already_dubbed_rows)


# ---------------------------------------------------------------------------
# Phase 2 - Video discovery and probing
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv", ".wmv", ".mpg", ".mpeg",
}

MAX_DURATION = 1799
MIN_WIDTH = 640
MIN_HEIGHT = 480

SCALE_HEIGHT = {"480p": 480, "720p": 720, "1080p": 1080}

# Split-segment pattern: <anything>_<two digits><ext>
SPLIT_SEGMENT_RE = re.compile(r"^(?P<base>.+)_(?P<num>\d{2})$")


def list_videos(folder: Path) -> list:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def _run_ffprobe(path: Path, args: list) -> Optional[str]:
    cmd = ["ffprobe", "-v", "error", *args, str(path)]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, errors="replace", check=True,
        )
        return out.stdout.strip() if out.stdout else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_duration(path: Path) -> Optional[float]:
    raw = _run_ffprobe(
        path, ["-show_entries", "format=duration", "-of", "default=nw=1:nk=1"]
    )
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def get_resolution(path: Path) -> Optional[tuple]:
    raw = _run_ffprobe(
        path,
        [
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
        ],
    )
    if not raw or "," not in raw:
        return None
    try:
        parts = raw.splitlines()[0].split(",")
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return None


def get_bit_rate(path: Path) -> Optional[int]:
    """Overall container bitrate in bps. None if ffprobe cannot determine it
    (some containers omit it; we then derive it from filesize/duration)."""
    raw = _run_ffprobe(
        path,
        ["-show_entries", "format=bit_rate", "-of", "default=nw=1:nk=1"],
    )
    if raw and raw != "N/A":
        try:
            n = int(raw)
            if n > 0:
                return n
        except ValueError:
            pass
    # Fallback: compute from filesize / duration
    try:
        size = path.stat().st_size
        dur = get_duration(path)
        if size and dur and dur > 0:
            return int(size * 8 / dur)
    except OSError:
        pass
    return None


def probe_video(path: Path) -> dict:
    dur = get_duration(path)
    wh = get_resolution(path)
    return {
        "path": path,
        "duration": dur,
        "width": wh[0] if wh else None,
        "height": wh[1] if wh else None,
        "bit_rate": get_bit_rate(path),
        "ok": dur is not None and wh is not None,
    }


# Per-resolution bitrate threshold. Below this, re-encoding with H.265 will
# typically NOT shrink the file (the source is already at or under what x265
# CRF 22 would output for that resolution). We pick the threshold by source
# height: 480p / 720p / 1080p / 4K.
def compress_threshold_bps(height: Optional[int]) -> int:
    if height is None:
        return 1_500_000
    if height <= 540:
        return 1_000_000
    if height <= 800:
        return 1_500_000
    if height <= 1200:
        return 3_000_000
    return 10_000_000


def should_compress_video(probe: dict, settings: dict) -> bool:
    """True if THIS video is worth re-encoding for compression purposes.
    Compression is enabled by the user AND the source bitrate is above the
    threshold for its resolution."""
    if not settings.get("compress"):
        return False
    bit_rate = probe.get("bit_rate")
    if bit_rate is None:
        return False  # cannot decide, leave as-is
    return bit_rate >= compress_threshold_bps(probe.get("height"))


def probe_all(videos: list) -> list:
    results = []
    total = len(videos)
    for i, v in enumerate(videos, 1):
        # \033[2K erases the whole line, \r returns the cursor to column 0.
        print(
            f"\033[2K\r  {C.DIM}[{i}/{total}] probing {v.name}...{C.R}",
            end="",
            flush=True,
        )
        results.append(probe_video(v))
    print("\033[2K\r", end="")  # clear the progress line once finished
    return results


# ---------------------------------------------------------------------------
# Phase 2 - Output path resolution
# ---------------------------------------------------------------------------

def _target_extension(input_path: Path, container: Optional[str]) -> str:
    return container if container else input_path.suffix


def build_output_path(
    input_path: Path,
    policy: dict,
    container: Optional[str] = None,
    segment: Optional[tuple] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Compose the final output path from input + settings.

    segment:    (index, total) if the output is a split segment, else None.
                Naming: <stem><suffix>_NN<ext> with NN as two digits.
    policy:     {"overwrite": bool, "suffix": str}. An empty suffix means overwrite.
    output_dir: optional folder for ALL outputs. None means each output goes
                next to its source video.
    """
    ext = _target_extension(input_path, container)
    suffix = policy.get("suffix", "")
    base = input_path.stem + suffix
    if segment is not None:
        idx, _total = segment
        base = f"{base}_{idx:02d}"
    parent = output_dir if output_dir is not None else input_path.parent
    return parent / f"{base}{ext}"


def detect_split_groups(videos: list) -> dict:
    """
    Group videos whose stems match <base>_NN. Returns:
      {base_name: [(path, segment_number), ...]} sorted by segment number.

    Bases that only have a single member are excluded (nothing to join).
    """
    groups: dict = {}
    for v in videos:
        m = SPLIT_SEGMENT_RE.match(v.stem)
        if not m:
            continue
        base = m.group("base")
        num = int(m.group("num"))
        groups.setdefault(base, []).append((v, num))
    return {
        base: sorted(items, key=lambda x: x[1])
        for base, items in groups.items()
        if len(items) >= 2
    }


# ---------------------------------------------------------------------------
# Phase 3c - Encoder autodetection (GPU if available, CPU otherwise)
# ---------------------------------------------------------------------------
# ffmpeg ships with several hardware encoders (NVENC, AMF, QSV, VideoToolbox)
# and the software fallbacks libx265 / libx264. We pick the best one per
# machine at startup by (a) listing compiled encoders and (b) running a tiny
# dummy encode with each candidate to verify the GPU driver actually works -
# a compiled encoder is not enough. The final choice is stored in a module
# global that build_ffmpeg_cmd consults for every output.

_ENCODER_PRIORITY_H265 = [
    "hevc_nvenc", "hevc_amf", "hevc_qsv", "hevc_videotoolbox", "libx265",
]
_ENCODER_PRIORITY_H264 = [
    "h264_nvenc", "h264_amf", "h264_qsv", "h264_videotoolbox", "libx264",
]

_ENCODER_LABEL = {
    "hevc_nvenc": "NVIDIA GPU (NVENC, H.265)",
    "h264_nvenc": "NVIDIA GPU (NVENC, H.264)",
    "hevc_amf":   "AMD GPU (AMF, H.265)",
    "h264_amf":   "AMD GPU (AMF, H.264)",
    "hevc_qsv":   "Intel GPU (QuickSync, H.265)",
    "h264_qsv":   "Intel GPU (QuickSync, H.264)",
    "hevc_videotoolbox": "Apple VideoToolbox (H.265)",
    "h264_videotoolbox": "Apple VideoToolbox (H.264)",
    "libx265":    "CPU (libx265, H.265)",
    "libx264":    "CPU (libx264, H.264)",
}


# Populated at startup by detect_encoders(). Consulted by build_ffmpeg_cmd().
DETECTED = {
    "h265_encoder": "libx265",
    "h264_encoder": "libx264",
    "gpu_available": False,
    "gpu_enabled": False,
}


def _list_ffmpeg_encoders() -> set:
    """Return the set of encoder names compiled into the local ffmpeg."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        return set()
    names = set()
    for line in r.stdout.splitlines():
        # Encoder lines look like:  " V..... libx264 ... description ..."
        stripped = line.strip()
        if stripped.startswith("V"):
            parts = stripped.split(None, 2)
            if len(parts) >= 2:
                names.add(parts[1])
    return names


def _probe_encoder(name: str) -> bool:
    """Run a 0.1s dummy encode to verify the encoder actually works."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:s=256x256:r=30:d=0.1",
        "-pix_fmt", "yuv420p",
        "-c:v", name,
        "-f", "null", "-",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return r.returncode == 0
    except Exception:
        return False


def _first_working_encoder(candidates: list, available: set) -> Optional[str]:
    for name in candidates:
        if name in available and _probe_encoder(name):
            return name
    return None


def detect_encoders() -> None:
    """Populate the DETECTED global with the best available encoders."""
    available = _list_ffmpeg_encoders()

    h265 = _first_working_encoder(_ENCODER_PRIORITY_H265, available) or "libx265"
    h264 = _first_working_encoder(_ENCODER_PRIORITY_H264, available) or "libx264"

    gpu_available = h265 != "libx265" or h264 != "libx264"

    DETECTED["h265_encoder"] = h265
    DETECTED["h264_encoder"] = h264
    DETECTED["gpu_available"] = gpu_available
    DETECTED["gpu_enabled"] = gpu_available  # default on if available


def _encoder_args(encoder: str, crf: int) -> list:
    """Map our universal CRF-ish value to the encoder's specific flags."""
    if encoder == "libx265":
        # log-level=error stops libx265 from interleaving its own status
        # messages with ffmpeg's stats line, which otherwise turns the
        # progress display into a wall of text instead of a single
        # \r-overwriting line.
        return [
            "-c:v", "libx265", "-crf", str(crf), "-preset", "medium",
            "-x265-params", "log-level=error",
            "-tag:v", "hvc1",
        ]
    if encoder == "libx264":
        return [
            "-c:v", "libx264", "-crf", str(crf), "-preset", "fast",
            "-x264-params", "log_level=error",
        ]
    if encoder == "hevc_nvenc":
        return [
            "-c:v", "hevc_nvenc", "-preset", "p5",
            "-rc", "vbr", "-cq", str(crf), "-b:v", "0",
            "-tag:v", "hvc1",
        ]
    if encoder == "h264_nvenc":
        return [
            "-c:v", "h264_nvenc", "-preset", "p5",
            "-rc", "vbr", "-cq", str(crf), "-b:v", "0",
        ]
    if encoder == "hevc_amf":
        return [
            "-c:v", "hevc_amf", "-quality", "balanced",
            "-rc", "cqp", "-qp_i", str(crf), "-qp_p", str(crf),
            "-tag:v", "hvc1",
        ]
    if encoder == "h264_amf":
        return [
            "-c:v", "h264_amf", "-quality", "balanced",
            "-rc", "cqp", "-qp_i", str(crf), "-qp_p", str(crf),
        ]
    if encoder == "hevc_qsv":
        return [
            "-c:v", "hevc_qsv", "-preset", "medium",
            "-global_quality", str(crf),
            "-tag:v", "hvc1",
        ]
    if encoder == "h264_qsv":
        return [
            "-c:v", "h264_qsv", "-preset", "medium",
            "-global_quality", str(crf),
        ]
    if encoder == "hevc_videotoolbox":
        # VideoToolbox uses 1..100 (100 = best). Map crf 18..32 -> ~80..50.
        q = max(1, min(100, 100 - (crf - 18) * 2))
        return ["-c:v", "hevc_videotoolbox", "-q:v", str(q), "-tag:v", "hvc1"]
    if encoder == "h264_videotoolbox":
        q = max(1, min(100, 100 - (crf - 18) * 2))
        return ["-c:v", "h264_videotoolbox", "-q:v", str(q)]
    return ["-c:v", encoder]


def pick_encoder(codec: str) -> str:
    """Return the encoder name to use for 'h265' or 'h264' right now."""
    if codec == "h265":
        return DETECTED["h265_encoder"] if DETECTED["gpu_enabled"] else "libx265"
    return DETECTED["h264_encoder"] if DETECTED["gpu_enabled"] else "libx264"


# ---------------------------------------------------------------------------
# Phase 3a - Silence detection and cut strategy
# ---------------------------------------------------------------------------
# When a video exceeds MAX_DURATION, we split it at the best silent moments
# and optionally speed it up (<= MAX_SPEED_FACTOR) to keep each piece under
# MAX_DURATION. The cut-point search is identical in spirit to the original
# standalone script: try every detected pause within the valid window and
# pick the one whose effective pause (after applying the minimum required
# speed-up) is longest.

MIN_SILENCE_DURATION = 1.0
SILENCE_THRESHOLD_DB = -35
MAX_SPEED_FACTOR = 1.10
MAX_FRAG_SPEED = MAX_DURATION * MAX_SPEED_FACTOR  # ~1978.9 seconds


def estimate_segments(duration: float) -> int:
    """Cheap estimate - used in preview to avoid running the full analysis."""
    if duration <= MAX_DURATION:
        return 1
    return max(2, math.ceil(duration / MAX_FRAG_SPEED))


def detect_silences_ffmpeg(path: Path) -> list:
    cmd = [
        "ffmpeg", "-i", str(path),
        "-af", f"silencedetect=noise={SILENCE_THRESHOLD_DB}dB:d={MIN_SILENCE_DURATION}",
        "-f", "null", "-",
    ]
    try:
        # errors="replace" is required: ffmpeg dumps the input's metadata to
        # stderr, and on Windows the default cp1252 codec blows up the
        # subprocess reader thread on any byte outside its tiny range (e.g.
        # a smart quote in a Title tag), leaving res.stderr=None.
        res = subprocess.run(
            cmd, capture_output=True, text=True, errors="replace",
        )
    except FileNotFoundError:
        return []

    if not res.stderr:
        return []
    silences = []
    current = {}
    for line in res.stderr.splitlines():
        if "silence_start" in line:
            try:
                current["start"] = float(line.split("silence_start:")[1].strip())
            except (IndexError, ValueError):
                current = {}
        elif "silence_end" in line:
            try:
                parts = line.split("|")
                current["end"] = float(parts[0].split("silence_end:")[1].strip())
                current["duration"] = float(parts[1].split("silence_duration:")[1].strip())
                silences.append(dict(current))
            except (IndexError, ValueError):
                pass
            current = {}
    return silences


_vad_cache: Optional[tuple] = None


def _load_vad():
    """Lazy-load silero-vad once per process. Returns None if unavailable."""
    global _vad_cache
    if _vad_cache is not None:
        return _vad_cache
    try:
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            import torch  # noqa: F401
            model, utils = __import__("torch").hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )
        _vad_cache = (model, utils)
        return _vad_cache
    except Exception:
        _vad_cache = ()  # sentinel so we do not retry
        return None


def detect_silences_vad(path: Path) -> list:
    """Return VAD-detected pauses. Empty list if VAD is unavailable or fails."""
    loaded = _load_vad()
    if not loaded:
        return []
    model, utils = loaded
    get_speech_timestamps, _, read_audio, *_ = utils

    import tempfile as _tf
    tmp_wav = Path(_tf.mkstemp(suffix=".wav", prefix="vad_")[1])
    try:
        conv = subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-ar", "16000", "-ac", "1", str(tmp_wav)],
            capture_output=True,
        )
        if conv.returncode != 0:
            return []

        wav = read_audio(str(tmp_wav), sampling_rate=16000)
        sr = 16000
        speech_ts = get_speech_timestamps(wav, model, sampling_rate=sr)
        speech = [(s["start"] / sr, s["end"] / sr) for s in speech_ts]

        pauses = []
        prev_end = 0.0
        for start, end in speech:
            gap = start - prev_end
            if gap >= MIN_SILENCE_DURATION:
                pauses.append({"start": prev_end, "end": start, "duration": gap})
            prev_end = end

        total = len(wav) / sr
        if total - prev_end >= MIN_SILENCE_DURATION:
            pauses.append({"start": prev_end, "end": total, "duration": total - prev_end})
        return pauses
    except Exception:
        return []
    finally:
        try:
            tmp_wav.unlink(missing_ok=True)
        except Exception:
            pass


def merge_pauses(
    ffmpeg_pauses: list, vad_pauses: list, tolerance: float = 1.0
) -> list:
    """Merge two pause lists; when two are within `tolerance`, keep the longer."""
    merged = []
    for p in sorted(ffmpeg_pauses + vad_pauses, key=lambda x: x["start"]):
        if merged and abs(p["start"] - merged[-1]["start"]) < tolerance:
            if p["duration"] > merged[-1]["duration"]:
                merged[-1] = p
        else:
            merged.append(p)
    return merged


def calculate_strategy(duration: float, pauses: list) -> dict:
    """
    Decide how to partition a long video. Output dict shape:
      {
        "fragments":      int,
        "speed_factor":   float,  # max speed across all segments
        "segment_speeds": [float, ...],
        "cut_points":     [{"cut_at_seconds": float, ...}, ...],
        "strategy":       "no_cut" | "direct_cut" | "speed_up",
        "description":    str,
      }
    """
    def eval_pause(pause_duration: float, frag_duration: float):
        if frag_duration <= MAX_DURATION:
            return pause_duration, 1.0
        if frag_duration <= MAX_FRAG_SPEED:
            speed = frag_duration / MAX_DURATION
            return pause_duration / speed, round(speed, 4)
        return -1, None

    if duration <= MAX_DURATION:
        return {
            "fragments": 1,
            "speed_factor": 1.0,
            "segment_speeds": [1.0],
            "cut_points": [],
            "strategy": "no_cut",
            "description": "Fits whole, no cutting required.",
        }

    n_fragments = max(2, math.ceil(duration / MAX_FRAG_SPEED))
    cut_points = []
    segment_speeds = []
    prev_cut = 0.0

    for i in range(1, n_fragments):
        remaining = n_fragments - i
        min_cut = max(prev_cut + 1.0, duration - remaining * MAX_FRAG_SPEED)
        max_cut = prev_cut + MAX_FRAG_SPEED

        best_pause = None
        best_eff = -1.0
        best_speed = 1.0
        for p in pauses:
            cut_at = (p["start"] + p["end"]) / 2.0
            if cut_at < min_cut or cut_at > max_cut:
                continue
            frag_dur = cut_at - prev_cut
            eff, spd = eval_pause(p["duration"], frag_dur)
            if eff > best_eff:
                best_eff = eff
                best_pause = p
                best_speed = spd

        if best_pause is not None:
            cut_at = (best_pause["start"] + best_pause["end"]) / 2.0
            cut_points.append({
                "cut_at_seconds": round(cut_at, 2),
                "pause_duration": round(best_pause["duration"], 2),
                "pause_start": round(best_pause["start"], 2),
                "pause_end": round(best_pause["end"], 2),
                "effective_pause": round(best_eff, 2),
                "speed_factor": round(best_speed, 4),
            })
            segment_speeds.append(round(best_speed, 4))
            prev_cut = cut_at
        else:
            fallback = (min_cut + max_cut) / 2.0
            cut_points.append({
                "cut_at_seconds": round(fallback, 2),
                "pause_duration": 0,
                "pause_start": None,
                "pause_end": None,
                "effective_pause": 0,
                "speed_factor": 1.0,
                "warning": "No valid pause found - forced cut",
            })
            segment_speeds.append(1.0)
            prev_cut = fallback

    last_dur = duration - prev_cut
    if last_dur > MAX_DURATION:
        last_speed = (
            round(last_dur / MAX_DURATION, 4)
            if last_dur <= MAX_FRAG_SPEED
            else MAX_SPEED_FACTOR
        )
    else:
        last_speed = 1.0
    segment_speeds.append(last_speed)

    max_speed = max(segment_speeds)
    if max_speed > 1.001:
        desc = (f"Split into {n_fragments} segment(s), max speed-up "
                f"{round((max_speed - 1) * 100, 1)}%")
        strategy = "speed_up"
    else:
        desc = f"Split into {n_fragments} segment(s), no speed-up"
        strategy = "direct_cut"

    return {
        "fragments": n_fragments,
        "speed_factor": round(max_speed, 4),
        "segment_speeds": segment_speeds,
        "cut_points": cut_points,
        "strategy": strategy,
        "description": desc,
    }


def analyze_long_video(path: Path) -> dict:
    """Full silence analysis + strategy for one long video. Returns the strategy dict."""
    dur = get_duration(path) or 0.0
    ff_pauses = detect_silences_ffmpeg(path)
    vad_pauses = detect_silences_vad(path)
    pauses = merge_pauses(ff_pauses, vad_pauses)
    strategy = calculate_strategy(dur, pauses)
    strategy["_duration"] = dur
    strategy["_pauses_ffmpeg"] = len(ff_pauses)
    strategy["_pauses_vad"] = len(vad_pauses)
    strategy["_pauses_total"] = len(pauses)
    return strategy


def analyze_long_videos(probes: list) -> dict:
    """Run analysis for every video over MAX_DURATION. Returns {Path: strategy}."""
    long_ones = [p for p in probes if p["ok"] and (p["duration"] or 0) > MAX_DURATION]
    if not long_ones:
        return {}
    head(f"Analysing {len(long_ones)} long video(s)")
    info("Silence detection can take a few minutes per video. Ctrl+C to skip.")
    results: dict = {}
    total = len(long_ones)
    for i, probe in enumerate(long_ones, 1):
        print(
            f"\033[2K\r  {C.DIM}[{i}/{total}] analysing {probe['path'].name}...{C.R}",
            end="",
            flush=True,
        )
        try:
            results[probe["path"]] = analyze_long_video(probe["path"])
        except KeyboardInterrupt:
            print("\033[2K\r", end="")
            warn("Analysis interrupted - remaining videos will be analysed on the fly")
            break
        except Exception as e:
            results[probe["path"]] = {"_error": str(e)}
    print("\033[2K\r", end="")
    return results


# ---------------------------------------------------------------------------
# Preview (what would happen) - replaces actual execution
# ---------------------------------------------------------------------------

def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fmt_resolution(p: dict) -> str:
    if p["width"] is None or p["height"] is None:
        return "?x?"
    return f"{p['width']}x{p['height']}"


def _fmt_bitrate(bps: Optional[int]) -> str:
    if not bps:
        return "?"
    if bps >= 1_000_000:
        return f"{bps / 1_000_000:.1f}M"
    return f"{bps // 1000}k"


def preview(folder: Path, settings: dict) -> tuple:
    """
    Run discovery + probing + planned-action preview.

    Returns (proceed: bool, probes: list, analyses: dict).
    analyses maps Path -> strategy dict for any long video that was
    analysed in advance (detailed cut plan). Phase 3b will analyse any
    long videos missing here on the fly during execution.
    """
    head("Scanning folder")
    info(f"Folder: {folder}")
    videos = list_videos(folder)
    if not videos:
        bad("No video files found in that folder.")
        return False, [], {}
    ok(f"Found {len(videos)} video file(s).")

    probes = probe_all(videos)
    unreadable = [p for p in probes if not p["ok"]]
    for p in unreadable:
        warn(f"Could not probe: {p['path'].name}")

    readable = [p for p in probes if p["ok"]]
    if not readable:
        bad("None of the videos could be read by ffprobe. Aborting.")
        return False, [], {}

    head("Inventory")
    print(
        f"  {'File':<40} {'Resolution':>12} {'Bitrate':>9} "
        f"{'Duration':>10} {'Status':>8}"
    )
    print(
        f"  {'-' * 40} {'-' * 12} {'-' * 9} "
        f"{'-' * 10} {'-' * 8}"
    )
    for p in probes:
        name = p["path"].name
        if len(name) > 39:
            name = name[:36] + "..."
        if not p["ok"]:
            status, status_color = "unread", C.RED
        elif already_processed_marker(p, settings):
            status, status_color = "skip", C.DIM
        elif settings["mode"] == "to_dub" and not needs_any_work(p, settings):
            status, status_color = "passthru", C.DIM
        else:
            status, status_color = "ok", C.G
        print(
            f"  {name:<40} {_fmt_resolution(p):>12} "
            f"{_fmt_bitrate(p.get('bit_rate')):>9} "
            f"{_fmt_duration(p['duration']):>10} "
            f"{status_color}{status:>8}{C.R}"
        )
    print()

    # From here on we keep only files we will actually touch:
    # - drop unreadable
    # - drop files matching the active suffix (cascade guard)
    # - in to_dub mode, drop files where needs_any_work is False
    workable = [
        p for p in readable
        if not already_processed_marker(p, settings)
        and (settings["mode"] != "to_dub" or needs_any_work(p, settings))
    ]
    skipped = len(readable) - len(workable)
    if skipped:
        info(
            f"{skipped} readable file(s) need no work and will be left alone."
        )
    readable = workable
    if not readable:
        warn("Nothing to do - no file actually needs any of the selected ops.")
        return False, probes, {}

    head("Planned actions")
    steps = []
    long_videos = []
    compress_targets = []

    if settings["mode"] == "to_dub":
        if settings["resize_small"]:
            n = sum(
                1 for p in readable
                if (p["width"] or 0) < MIN_WIDTH or (p["height"] or 0) < MIN_HEIGHT
            )
            steps.append(f"Resize videos under 640x480 ({n} match)")
        if settings["scale_down"]:
            target_h = SCALE_HEIGHT[settings["scale_down"]]
            n = sum(1 for p in readable if (p["height"] or 0) > target_h)
            steps.append(f"Scale down to {settings['scale_down']} ({n} match)")
        if settings["process_long"]:
            long_videos = [
                p for p in readable if (p["duration"] or 0) > MAX_DURATION
            ]
            est_segments = sum(estimate_segments(p["duration"]) for p in long_videos)
            steps.append(
                f"Split / speed up videos over 1799s "
                f"({len(long_videos)} match -> ~{est_segments} output segments)"
            )
    else:
        if settings["join"]:
            groups = detect_split_groups([p["path"] for p in readable])
            total_segments = sum(len(v) for v in groups.values())
            steps.append(
                f"Join split segments ({len(groups)} group(s), "
                f"{total_segments} segment(s))"
            )

    if settings["compress"]:
        compress_targets = [p for p in readable if should_compress_video(p, settings)]
        steps.append(
            f"Compress with H.265 CRF {settings['quality_crf']} "
            f"({len(compress_targets)} of {len(readable)} above bitrate threshold)"
        )

    if settings["container"]:
        steps.append(f"Change container to {settings['container']}")

    policy = settings["policy"]
    if policy["overwrite"]:
        steps.append("Overwrite originals on success")
    else:
        steps.append(f'Save with suffix "{policy["suffix"]}"')

    if not steps:
        warn("Nothing to do - every option is off.")
        return False, probes, {}

    for i, step in enumerate(steps, 1):
        print(f"    {i}. {step}")

    # Sample output paths so the user sees the naming in action.
    head("Sample output paths")
    if settings["mode"] == "already_dubbed":
        # The unit of work is a group, not a file. Show grouped previews
        # first, then a couple of standalones if any.
        readable_paths = [p["path"] for p in readable]
        groups = (
            detect_split_groups(readable_paths) if settings.get("join") else {}
        )
        grouped_paths = {
            path for items in groups.values() for path, _ in items
        }
        standalones = [p for p in readable if p["path"] not in grouped_paths]

        shown = 0
        for base, items in list(groups.items())[:3]:
            seg_paths = [path for path, _ in items]
            first = seg_paths[0]
            pseudo_input = first.with_name(f"{base}{first.suffix}")
            out = build_output_path(
                pseudo_input, policy,
                container=settings["container"],
                output_dir=settings.get("output_dir"),
            )
            print(
                f"    {C.DIM}{base}  ({len(seg_paths)} segments){C.R}  ->  "
                f"{out.name}"
            )
            shown += 1
        if len(groups) > 3:
            print(f"    {C.DIM}... and {len(groups) - 3} more group(s){C.R}")

        do_standalones = bool(
            settings.get("compress") or settings.get("container")
        )
        if do_standalones and standalones:
            print()
            for p in standalones[:2]:
                out = build_output_path(
                    p["path"], policy,
                    container=settings["container"],
                    output_dir=settings.get("output_dir"),
                )
                print(f"    {p['path'].name}  ->  {out.name}")
            if len(standalones) > 2:
                print(
                    f"    {C.DIM}... and {len(standalones) - 2} more "
                    f"standalone(s){C.R}"
                )
    else:
        samples = readable[:3]
        for p in samples:
            dur = p["duration"] or 0
            will_split = (
                settings["process_long"]
                and dur > MAX_DURATION
            )
            est_n = estimate_segments(dur) if will_split else 1
            seg = (1, est_n) if will_split and est_n > 1 else None
            out = build_output_path(
                p["path"], policy,
                container=settings["container"],
                segment=seg,
                output_dir=settings.get("output_dir"),
            )
            tag = f" (first of ~{est_n} segments)" if seg else ""
            print(f"    {p['path'].name}  ->  {out.name}{tag}")
        if len(readable) > 3:
            print(f"    {C.DIM}... and {len(readable) - 3} more{C.R}")

    # Optional: run the full silence analysis now so the user can review
    # the exact cut plan before committing to execution.
    analyses: dict = {}
    if long_videos:
        print()
        info(
            f"{len(long_videos)} video(s) will need silence analysis. "
            "Running it now gives you an exact cut"
        )
        print("       plan; skipping defers it to execution time (same total cost).")
        if ask_yes_no("Run silence analysis now to preview the cut plan?", default="N"):
            analyses = analyze_long_videos(long_videos)
            _print_analyses(analyses, settings, policy)

    print()
    proceed = ask_yes_no("Proceed?", default="Y")
    return proceed, probes, analyses


def _print_analyses(analyses: dict, settings: dict, policy: dict) -> None:
    if not analyses:
        return
    head("Cut plan")
    for path, strat in analyses.items():
        if "_error" in strat:
            bad(f"{path.name}: analysis failed ({strat['_error']})")
            continue
        dur = strat.get("_duration", 0)
        print(
            f"  {C.B}{path.name}{C.R}  "
            f"{C.DIM}({_fmt_duration(dur)}, {strat['_pauses_total']} pauses){C.R}"
        )
        print(f"    -> {strat['description']}")
        if strat["strategy"] == "no_cut":
            continue
        boundaries = [0.0] + [cp["cut_at_seconds"] for cp in strat["cut_points"]] + [dur]
        for i in range(len(boundaries) - 1):
            a, b = boundaries[i], boundaries[i + 1]
            speed = strat["segment_speeds"][i]
            seg = (i + 1, strat["fragments"])
            out = build_output_path(
                path, policy,
                container=settings["container"],
                segment=seg,
                output_dir=settings.get("output_dir"),
            )
            print(
                f"    segment {i + 1}/{strat['fragments']}: "
                f"{_fmt_duration(a)}-{_fmt_duration(b)} "
                f"({_fmt_duration(b - a)}) "
                f"speed x{speed}  ->  {out.name}"
            )
        print()


# ---------------------------------------------------------------------------
# Phase 3b - Operation planner and single-pass ffmpeg execution
# ---------------------------------------------------------------------------
# For every video that needs work we build ONE ffmpeg invocation per output
# file. That invocation chains every required filter (scale, pad, setpts,
# atempo) and codec choice (libx264/libx265 + aac, or stream copy when
# nothing changes) so the file is encoded only once.

def needs_any_work(probe: dict, settings: dict) -> bool:
    """True if at least one configured operation will modify this video."""
    if settings["mode"] != "to_dub":
        return False
    w, h, dur = probe["width"] or 0, probe["height"] or 0, probe["duration"] or 0
    if settings["resize_small"] and (w < MIN_WIDTH or h < MIN_HEIGHT):
        return True
    if settings["scale_down"] and h > SCALE_HEIGHT[settings["scale_down"]]:
        return True
    if settings["process_long"] and dur > MAX_DURATION:
        return True
    if should_compress_video(probe, settings):
        return True
    if settings["container"]:
        return True
    return False


def already_processed_marker(probe: dict, settings: dict) -> Optional[str]:
    """Skip when the file's stem already ends in the active suffix - avoids
    cascading video_mod_mod_mod.mp4 names on re-runs. Files that simply do
    not need any work are filtered out elsewhere by needs_any_work, NOT
    here, because that decision belongs to the planner."""
    suffix = settings["policy"].get("suffix", "")
    if suffix and probe["path"].stem.endswith(suffix):
        return f'name already ends in "{suffix}"'
    return None


def plan_segments_for_video(
    probe: dict, settings: dict, analyses: dict
) -> list:
    """Decide the list of output segments for one video. Each item is a dict:
       {"t_start": float|None, "t_end": float|None, "speed": float,
        "out_index": int|None, "out_total": int}.
    out_index=None means a single output file with no _NN suffix.
    """
    dur = probe["duration"] or 0

    if settings["process_long"] and dur > MAX_DURATION:
        if dur <= MAX_FRAG_SPEED:
            speed = round(dur / MAX_DURATION, 4)
            return [{
                "t_start": None, "t_end": None, "speed": speed,
                "out_index": None, "out_total": 1,
            }]
        # Need to split. Use cached analysis if available, otherwise compute now.
        strat = analyses.get(probe["path"])
        if strat is None or "_error" in (strat or {}):
            strat = analyze_long_video(probe["path"])
        boundaries = (
            [0.0]
            + [cp["cut_at_seconds"] for cp in strat["cut_points"]]
            + [dur]
        )
        segs = []
        n = strat["fragments"]
        for i in range(n):
            segs.append({
                "t_start": boundaries[i],
                "t_end": boundaries[i + 1],
                "speed": strat["segment_speeds"][i],
                "out_index": i + 1,
                "out_total": n,
            })
        return segs

    return [{
        "t_start": None, "t_end": None, "speed": 1.0,
        "out_index": None, "out_total": 1,
    }]


def _vf_for(segment: dict, settings: dict, probe: dict) -> Optional[str]:
    parts = []
    w, h = probe["width"] or 0, probe["height"] or 0

    if settings["resize_small"] and (w < MIN_WIDTH or h < MIN_HEIGHT):
        parts.append(
            f"scale={MIN_WIDTH}:{MIN_HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={MIN_WIDTH}:{MIN_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1:1"
        )
    elif settings["scale_down"]:
        target = SCALE_HEIGHT[settings["scale_down"]]
        if h > target:
            parts.append(f"scale=-2:{target}")

    if abs(segment["speed"] - 1.0) > 0.001:
        parts.append(f"setpts=(1/{segment['speed']:.6f})*PTS")

    return ",".join(parts) if parts else None


def _af_for(segment: dict) -> Optional[str]:
    if abs(segment["speed"] - 1.0) > 0.001:
        return f"atempo={segment['speed']:.4f}"
    return None


def build_ffmpeg_cmd(
    input_path: Path, output_path: Path,
    segment: dict, settings: dict, probe: dict,
) -> list:
    """One ffmpeg invocation for one output file. Stream copy whenever no
    actual transformation is needed. Resize / scale / speed re-encodes use
    CPU libx264 (more efficient per bit than hardware encoders and fast
    enough for typical inputs). GPU is reserved for the H.265 compression
    path, where its speed advantage actually matters on large batches."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats"]

    if segment["t_start"] is not None:
        cmd += ["-ss", f"{segment['t_start']:.3f}"]
    if segment["t_end"] is not None:
        cmd += ["-to", f"{segment['t_end']:.3f}"]

    cmd += ["-i", str(input_path)]

    vf = _vf_for(segment, settings, probe)
    af = _af_for(segment)
    will_compress = should_compress_video(probe, settings)
    will_reencode_video = bool(vf) or will_compress
    crf = settings.get("quality_crf") or QUALITY_CRF_DEFAULT

    if vf:
        cmd += ["-vf", vf]
    if af:
        cmd += ["-af", af]

    if will_compress:
        # H.265 path - honour the GPU preference for speed on big batches.
        cmd += _encoder_args(pick_encoder("h265"), crf)
    elif will_reencode_video:
        # Resize / scale / speed re-encode - CPU libx264 always.
        cmd += _encoder_args("libx264", crf)
    else:
        cmd += ["-c:v", "copy"]

    # Audio re-encode is only required when an audio filter actually changes
    # the timing (atempo for speed-up). Everything else can stream-copy the
    # original audio, preserving its bitrate and saving encode time.
    if af:
        cmd += ["-c:a", "aac", "-b:a", "160k"]
    else:
        cmd += ["-c:a", "copy"]

    # -movflags +faststart was removed on purpose: it forces ffmpeg to do a
    # second pass over the finished file to move the moov atom to the top,
    # which adds noticeable time on larger outputs. The dubbing platform
    # re-processes videos on upload so having the moov at the top buys us
    # nothing here. Put it back if the output ever needs to stream directly
    # from a web server.
    cmd.append(str(output_path))
    return cmd


def _run_ffmpeg(cmd: list, output_path: Path) -> tuple:
    """Run ffmpeg and verify the output. Returns (ok: bool, error: str).

    stderr is INHERITED (not captured) so the user sees ffmpeg's own
    progress output in real time (`frame= fps= time= speed=` line). The
    trade-off is that we do not capture the exact error message for the
    final report - but any error is already visible to the user as it
    happens, which is far more useful on long encodes than a clean report.
    """
    try:
        result = subprocess.run(cmd)
    except FileNotFoundError:
        return False, "ffmpeg binary not found"
    except KeyboardInterrupt:
        return False, "interrupted by user"
    except Exception as e:
        return False, f"unexpected: {e}"

    if result.returncode != 0:
        return False, f"ffmpeg exited with code {result.returncode} (see output above)"
    if not output_path.exists() or output_path.stat().st_size == 0:
        return False, "ffmpeg returned 0 but the output is missing or empty"
    # ffprobe verification was removed on purpose: on Windows the just-
    # written file is often locked briefly by Defender's real-time scan,
    # which makes the ffprobe call block until the scan finishes and the
    # program looks hung. ffmpeg's exit code plus a non-empty output file
    # is reliable enough; if the file is malformed the user will notice
    # when the dubbing platform rejects it.
    return True, ""


def _describe_segment_op(segment: dict, settings: dict, probe: dict) -> str:
    """Short human-readable description of what the ffmpeg pass will do."""
    w, h = probe["width"] or 0, probe["height"] or 0
    parts = []
    if settings["resize_small"] and (w < MIN_WIDTH or h < MIN_HEIGHT):
        parts.append(f"upscale {w}x{h} -> 640x480")
    elif settings["scale_down"] and h > SCALE_HEIGHT[settings["scale_down"]]:
        parts.append(f"scale down to {settings['scale_down']}")
    if segment["t_start"] is not None or segment["t_end"] is not None:
        start = segment["t_start"] or 0
        end = segment["t_end"] or probe["duration"] or 0
        dur = max(0.0, end - start)
        parts.append(f"cut {int(dur // 60)}:{int(dur % 60):02d}")
    if abs(segment["speed"] - 1.0) > 0.001:
        parts.append(f"speed x{segment['speed']}")
    if should_compress_video(probe, settings):
        encoder = pick_encoder("h265")
        parts.append(f"compress H.265 ({encoder})")
    if not parts:
        parts.append("remux (no re-encode)")
    return ", ".join(parts)


def _temp_for(final_path: Path) -> Path:
    """Sibling temp filename that ffmpeg can still infer the format from."""
    return final_path.with_name(f".tmp_{final_path.name}")


def process_video_to_dub(
    probe: dict, settings: dict, analyses: dict, policy: dict,
) -> dict:
    """Process one video. Returns:
       {"status": "ok"|"skipped"|"error",
        "outputs": [Path...], "error": str, "reason": str}
    """
    skip_reason = already_processed_marker(probe, settings)
    if skip_reason:
        return {"status": "skipped", "outputs": [], "error": "", "reason": skip_reason}
    if not needs_any_work(probe, settings):
        return {"status": "skipped", "outputs": [], "error": "",
                "reason": "no configured op applies to this video"}

    input_path = probe["path"]
    segments = plan_segments_for_video(probe, settings, analyses)

    temp_paths: list = []
    final_paths: list = []

    try:
        for seg in segments:
            seg_arg = (
                (seg["out_index"], seg["out_total"])
                if seg["out_index"] is not None
                else None
            )
            final_path = build_output_path(
                input_path, policy,
                container=settings["container"],
                segment=seg_arg,
                output_dir=settings.get("output_dir"),
            )
            temp_path = _temp_for(final_path)
            temp_path.unlink(missing_ok=True)  # leftover from a previous crash

            op_desc = _describe_segment_op(seg, settings, probe)
            if len(segments) > 1 and seg.get("out_index"):
                print(
                    f"  {C.DIM}  segment {seg['out_index']}/{seg['out_total']}: "
                    f"{op_desc}{C.R}"
                )
            else:
                print(f"  {C.DIM}  {op_desc}{C.R}")

            cmd = build_ffmpeg_cmd(input_path, temp_path, seg, settings, probe)
            ok_, err = _run_ffmpeg(cmd, temp_path)
            if not ok_:
                for t in temp_paths:
                    t.unlink(missing_ok=True)
                temp_path.unlink(missing_ok=True)
                return {"status": "error", "outputs": [], "error": err, "reason": ""}

            temp_paths.append(temp_path)
            final_paths.append(final_path)

        # All segments produced. Promote them atomically.
        if policy["overwrite"]:
            try:
                input_path.unlink(missing_ok=True)
            except Exception as e:
                # Don't fail the whole video - keep temps for the user to recover.
                return {"status": "error", "outputs": [],
                        "error": f"could not delete original: {e}", "reason": ""}

        for tmp, final in zip(temp_paths, final_paths):
            if final.exists() and final != input_path:
                final.unlink()
            tmp.replace(final)

        return {"status": "ok", "outputs": final_paths, "error": "", "reason": ""}

    except KeyboardInterrupt:
        for t in temp_paths:
            t.unlink(missing_ok=True)
        raise
    except Exception as e:
        for t in temp_paths:
            try:
                t.unlink(missing_ok=True)
            except Exception:
                pass
        return {"status": "error", "outputs": [], "error": str(e), "reason": ""}


def execute_to_dub(probes: list, settings: dict, analyses: dict) -> dict:
    """Run the full To-Dub batch. Returns a counters/error-log dict."""
    readable = [p for p in probes if p["ok"]]
    total = len(readable)
    counters = {"processed": 0, "skipped": 0, "errors": 0, "error_log": []}

    head("Executing")
    info("Each video is encoded in a single pass per output segment.")
    print()

    for i, probe in enumerate(readable, 1):
        print(
            f"  {C.B}[{i}/{total}] {probe['path'].name}{C.R}    "
            f"{C.DIM}(Press Ctrl+C to Skip this video or to Exit){C.R}"
        )
        try:
            res = process_video_to_dub(probe, settings, analyses, settings["policy"])
        except KeyboardInterrupt:
            # ffmpeg is already dead at this point (Ctrl+C was forwarded to
            # the child). Ask whether to abort the whole batch or just skip
            # this one video and keep going with the rest.
            if confirm_exit_or_skip():
                warn("Exiting batch by user request.")
                break
            warn(f"Skipping {probe['path'].name}, continuing with the next video.")
            counters["errors"] += 1
            counters["error_log"].append(
                (probe["path"].name, "interrupted by user")
            )
            print()
            continue

        if res["status"] == "ok":
            counters["processed"] += 1
            for out in res["outputs"]:
                ok(f"  -> {out.name}")
        elif res["status"] == "skipped":
            counters["skipped"] += 1
            print(f"  {C.DIM}  skipped: {res['reason']}{C.R}")
        else:
            counters["errors"] += 1
            counters["error_log"].append((probe["path"].name, res["error"]))
            bad(f"  failed: {res['error'][:200]}")
        print()

    return counters


# ---------------------------------------------------------------------------
# Subtitle merge (used during the Already-Dubbed join)
# ---------------------------------------------------------------------------
# When the user joins a group of video segments, sibling subtitle files with
# the same stem are detected and merged with timestamp shifting (each
# subtitle's timestamps are pushed forward by the cumulative duration of
# every preceding video segment). This produces a single .srt / .vtt that
# stays in sync with the merged video without inserting empty placeholder
# entries. ASS support is intentionally out of scope for V1 - it has a
# different structure (Script Info / V4+ Styles / Events) and would need
# section-aware preservation.

SUBTITLE_EXTENSIONS = (".srt", ".vtt")
SUBTITLE_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "cp1251", "latin-1")
_SRT_TIME_RE = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")


def _detect_encoding(data: bytes) -> Optional[str]:
    """Use chardet or charset_normalizer when available to detect the byte
    string's encoding. Returns None if no detector is installed or the
    detector is not confident enough. Confidence threshold is conservative
    (0.7) to avoid wrong guesses on short files."""
    try:
        import chardet  # type: ignore
        result = chardet.detect(data)
        if (result and result.get("encoding")
                and result.get("confidence", 0) >= 0.7):
            return result["encoding"]
    except ImportError:
        pass
    try:
        from charset_normalizer import from_bytes  # type: ignore
        best = from_bytes(data).best()
        if best and best.encoding:
            return best.encoding
    except ImportError:
        pass
    return None


def _read_subtitle_file(path: Path) -> Optional[str]:
    """Decode a subtitle file. Fast path tries UTF-8 (with or without BOM),
    which covers ~95% of modern subs without paying the detection cost. If
    UTF-8 fails, hand the bytes to chardet / charset_normalizer (when
    available) for a best-guess encoding, and keep a manual CP1252 / CP1251
    / Latin-1 fallback chain for hosts without a detector. Returns the
    decoded text, or None if every attempt raised."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None

    for enc in ("utf-8-sig", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue

    candidates: list = []
    detected = _detect_encoding(raw)
    if detected:
        candidates.append(detected)
    candidates += [
        enc for enc in SUBTITLE_ENCODINGS
        if enc not in ("utf-8-sig", "utf-8") and enc != detected
    ]
    for enc in candidates:
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return None


def _parse_srt_time(s: str) -> Optional[int]:
    m = _SRT_TIME_RE.match(s.strip())
    if not m:
        return None
    h, mn, sec, ms = (int(x) for x in m.groups())
    return ((h * 60 + mn) * 60 + sec) * 1000 + ms


def _format_srt_time(ms: int, sep: str = ",") -> str:
    if ms < 0:
        ms = 0
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def parse_subtitle(text: str) -> list:
    """Parse an SRT or VTT body into [(start_ms, end_ms, body), ...].
    Tolerates VTT's WEBVTT header and either the SRT (',') or VTT ('.')
    timestamp separator."""
    text = text.lstrip("﻿").strip()
    if text.upper().startswith("WEBVTT"):
        # Drop blocks that do not contain a timestamp arrow (header,
        # NOTE blocks, STYLE blocks, REGION blocks).
        chunks = re.split(r"\n\s*\n", text)
        text = "\n\n".join(c for c in chunks if "-->" in c)

    entries = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [ln for ln in block.splitlines() if ln.strip() != ""]
        if not lines:
            continue
        timecode_idx = next(
            (i for i, ln in enumerate(lines) if "-->" in ln), -1
        )
        if timecode_idx < 0:
            continue
        m = re.match(
            r"\s*(\d+:\d+:\d+[,.]\d+)\s*-->\s*(\d+:\d+:\d+[,.]\d+)",
            lines[timecode_idx],
        )
        if not m:
            continue
        start_ms = _parse_srt_time(m.group(1))
        end_ms = _parse_srt_time(m.group(2))
        if start_ms is None or end_ms is None:
            continue
        body = "\n".join(lines[timecode_idx + 1:])
        entries.append((start_ms, end_ms, body))
    return entries


def merge_subtitles(
    sub_paths: list,
    segment_durations_s: list,
    output_path: Path,
    out_format: str,
) -> bool:
    """Merge sub_paths in order, shifting each one's timestamps by the
    cumulative duration of preceding video segments. out_format is 'srt' or
    'vtt'. Returns True on success."""
    all_entries = []
    cumulative_offset_ms = 0
    for sub_path, dur_s in zip(sub_paths, segment_durations_s):
        text = _read_subtitle_file(sub_path)
        if text is None:
            return False
        entries = parse_subtitle(text)
        for start, end, body in entries:
            all_entries.append((
                start + cumulative_offset_ms,
                end + cumulative_offset_ms,
                body,
            ))
        cumulative_offset_ms += int(round((dur_s or 0.0) * 1000))

    sep = "." if out_format == "vtt" else ","
    lines: list = []
    if out_format == "vtt":
        lines.append("WEBVTT")
        lines.append("")
    for i, (start, end, body) in enumerate(all_entries, 1):
        if out_format == "srt":
            lines.append(str(i))
        lines.append(f"{_format_srt_time(start, sep)} --> {_format_srt_time(end, sep)}")
        lines.append(body)
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return True


def find_subtitle_siblings(segment_paths: list) -> tuple:
    """Look for matching subtitle siblings (same stem, .srt or .vtt) for
    every segment. Returns (sub_paths, ext) when ALL segments have the
    same-format sibling, or (None, None) when partial / inconsistent /
    absent. SRT takes precedence over VTT when both are available."""
    for ext in SUBTITLE_EXTENSIONS:
        siblings = [p.with_suffix(ext) for p in segment_paths]
        if all(s.is_file() for s in siblings):
            return siblings, ext
    return None, None


# ---------------------------------------------------------------------------
# Phase 4 - Already-Dubbed batch (join + optional compress / container)
# ---------------------------------------------------------------------------
# When a video group ends in _01, _02, ... it can be joined back together.
# The fast path uses ffmpeg's concat demuxer with -c copy when every segment
# shares its codec / resolution / framerate / audio params. Otherwise we
# fall back to the concat *filter*, which decodes and re-encodes once across
# the whole timeline. The filter path is also taken when the user opted into
# H.265 compression - that way the join + compress happens in a single
# ffmpeg pass instead of two.

def _stream_signature(path: Path) -> Optional[tuple]:
    """Probe one file for the codec params that matter for concat demuxer
    compatibility. Returns a tuple suitable for equality, or None on error."""
    raw = _run_ffprobe(
        path,
        [
            "-show_entries",
            "stream=codec_type,codec_name,width,height,r_frame_rate,"
            "channels,sample_rate,pix_fmt",
            "-of", "json",
        ],
    )
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        return None
    return (
        video.get("codec_name"),
        video.get("width"),
        video.get("height"),
        video.get("r_frame_rate"),
        video.get("pix_fmt"),
        audio.get("codec_name") if audio else None,
        audio.get("channels") if audio else None,
        audio.get("sample_rate") if audio else None,
    )


def streams_compatible(paths: list) -> bool:
    """True when every path shares the codec / resolution / fps / audio
    layout - the prerequisite for concat demuxer to work without re-encode."""
    if len(paths) < 2:
        return True
    sigs = [_stream_signature(p) for p in paths]
    if any(s is None for s in sigs):
        return False
    return all(s == sigs[0] for s in sigs)


def _write_concat_list(paths: list, list_path: Path) -> None:
    """Write the file list ffmpeg's concat demuxer expects. Each entry must
    look like: file 'absolute/path' - single quotes around the path, with
    inner single quotes escaped as '\\''."""
    lines = []
    for p in paths:
        escaped = str(p.resolve()).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def concat_streamcopy(paths: list, output_path: Path) -> bool:
    """Run ffmpeg with the concat demuxer + -c copy. Returns True on success."""
    list_path = output_path.with_suffix(output_path.suffix + ".concat.txt")
    try:
        _write_concat_list(paths, list_path)
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats",
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            str(output_path),
        ]
        try:
            result = subprocess.run(cmd)
        except (FileNotFoundError, KeyboardInterrupt):
            raise
        if result.returncode != 0:
            return False
        if not output_path.exists() or output_path.stat().st_size == 0:
            return False
        # No ffprobe round-trip here on purpose: on Windows the just-written
        # file (often hundreds of MB for joined videos) can be locked by
        # Defender's real-time scan, which makes ffprobe block until the
        # scan finishes - looking like the program hung after the encode.
        # ffmpeg's exit code plus a non-zero file size is reliable enough.
        return True
    finally:
        list_path.unlink(missing_ok=True)


def concat_filter(
    paths: list, output_path: Path, settings: dict, probe: dict,
) -> bool:
    """Run ffmpeg with the concat *filter* and re-encode the joined output.
    Used when streams are incompatible OR when the user requested H.265
    compression (so we do join + compress in one ffmpeg invocation)."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats"]
    for p in paths:
        cmd += ["-i", str(p)]
    n = len(paths)
    chain = "".join(f"[{i}:v:0][{i}:a:0]" for i in range(n))
    chain += f"concat=n={n}:v=1:a=1[outv][outa]"
    cmd += ["-filter_complex", chain, "-map", "[outv]", "-map", "[outa]"]

    will_compress = should_compress_video(probe, settings)
    crf = settings.get("quality_crf") or QUALITY_CRF_DEFAULT
    if will_compress:
        cmd += _encoder_args(pick_encoder("h265"), crf)
    else:
        cmd += _encoder_args("libx264", crf)
    cmd += ["-c:a", "aac", "-b:a", "160k"]
    cmd.append(str(output_path))

    try:
        result = subprocess.run(cmd)
    except (FileNotFoundError, KeyboardInterrupt):
        raise
    if result.returncode != 0:
        return False
    if not output_path.exists() or output_path.stat().st_size == 0:
        return False
    # See the note in concat_streamcopy: skip the ffprobe verification to
    # avoid waiting on Windows Defender to release the freshly-written file.
    return True


def process_join_group(
    base_name: str, segment_paths: list, settings: dict, primary_probe: dict,
) -> dict:
    """Join one detected group into a single output file."""
    if not segment_paths:
        return {"status": "skipped", "outputs": [], "error": "",
                "reason": "empty group"}

    # Capture each segment's duration UP FRONT, before any file operation.
    # The subtitle merge later needs these durations to shift timestamps,
    # and the segments may be deleted in the overwrite path before the
    # subtitle code runs - in which case re-probing them returns None.
    segment_durations = [get_duration(p) or 0.0 for p in segment_paths]

    first = segment_paths[0]
    pseudo_input = first.with_name(f"{base_name}{first.suffix}")
    final_path = build_output_path(
        pseudo_input,
        settings["policy"],
        container=settings["container"],
        output_dir=settings.get("output_dir"),
    )
    temp_path = _temp_for(final_path)
    temp_path.unlink(missing_ok=True)

    will_compress = should_compress_video(primary_probe, settings)
    container_changing = bool(settings["container"]) and \
        settings["container"] != first.suffix

    # Probe stream compatibility once - it ffprobes every segment and was
    # being called twice per group otherwise (once to pick the path and
    # once for the description).
    use_filter = will_compress or container_changing
    compatible = False if use_filter else streams_compatible(segment_paths)

    desc_parts = [f"join {len(segment_paths)} segments"]
    if will_compress:
        desc_parts.append(f"compress H.265 ({pick_encoder('h265')})")
    elif compatible:
        desc_parts.append("stream copy")
    else:
        desc_parts.append("re-encode (libx264)")
    print(f"  {C.DIM}  {', '.join(desc_parts)}{C.R}")

    try:
        if use_filter or not compatible:
            ok_ = concat_filter(segment_paths, temp_path, settings, primary_probe)
        else:
            ok_ = concat_streamcopy(segment_paths, temp_path)
    except KeyboardInterrupt:
        temp_path.unlink(missing_ok=True)
        raise

    if not ok_:
        temp_path.unlink(missing_ok=True)
        return {"status": "error", "outputs": [],
                "error": "ffmpeg concat failed - see output above", "reason": ""}

    # Diagnostic prints for the post-encode stages so a hang reveals which
    # specific step is blocked instead of a silent dead screen. The Lsize
    # line ffmpeg emits ends in \n, so these print on fresh lines.
    print(f"  {C.DIM}  finalizing: rename...{C.R}", flush=True)
    if final_path.exists() and final_path not in segment_paths:
        final_path.unlink()
    temp_path.replace(final_path)

    # Source files queued for deletion - actual unlink happens at the end of
    # the whole batch in execute_already_dubbed, not here. Doing it inline
    # was causing multi-minute hangs when Windows Defender held the just-
    # written joined output file's directory while we tried to remove the
    # large neighbouring segment files. Deferring the cleanup means the
    # whole join batch finishes quickly and we sweep the deletes at the
    # end when the OS has settled.
    to_delete: list = []
    if settings["policy"]["overwrite"]:
        for seg in segment_paths:
            if seg != final_path:
                to_delete.append(seg)

    sub_output: Optional[Path] = None
    sub_error: Optional[str] = None
    if settings.get("merge_subs"):
        sib_paths, sib_ext = find_subtitle_siblings(segment_paths)
        if sib_paths is not None:
            print(
                f"  {C.DIM}  finalizing: merging "
                f"{len(sib_paths)} {sib_ext} files...{C.R}",
                flush=True,
            )
            sub_target = final_path.with_suffix(sib_ext)
            sub_temp = _temp_for(sub_target)
            sub_temp.unlink(missing_ok=True)
            try:
                ok_sub = merge_subtitles(
                    sib_paths, segment_durations, sub_temp,
                    out_format="vtt" if sib_ext == ".vtt" else "srt",
                )
            except Exception as e:
                ok_sub = False
                sub_error = f"subtitle merge failed: {e}"

            if ok_sub:
                if sub_target.exists() and sub_target not in sib_paths:
                    sub_target.unlink()
                sub_temp.replace(sub_target)
                sub_output = sub_target
                if settings["policy"]["overwrite"]:
                    for sib in sib_paths:
                        if sib != sub_target:
                            to_delete.append(sib)
            else:
                sub_temp.unlink(missing_ok=True)
                if sub_error is None:
                    sub_error = "subtitle merge failed (encoding or format)"

    return {
        "status": "ok",
        "outputs": [final_path],
        "error": "",
        "reason": "",
        "sub_output": sub_output,
        "sub_error": sub_error,
        "to_delete": to_delete,
    }


def _build_standalone_cmd(
    input_path: Path, output_path: Path, settings: dict, probe: dict,
) -> list:
    """Single ffmpeg pass for a standalone Already-Dubbed video. Either H.265
    re-encode (when worth it by bitrate) or pure -c copy remux when only the
    container is changing."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats",
           "-i", str(input_path)]
    will_compress = should_compress_video(probe, settings)
    if will_compress:
        crf = settings.get("quality_crf") or QUALITY_CRF_DEFAULT
        cmd += _encoder_args(pick_encoder("h265"), crf)
        cmd += ["-c:a", "aac", "-b:a", "160k"]
    else:
        cmd += ["-c", "copy"]
    cmd.append(str(output_path))
    return cmd


def process_standalone_already_dubbed(probe: dict, settings: dict) -> dict:
    """Standalone (not part of any join group) video. Skipped if no compress
    or container change applies to it."""
    will_compress = should_compress_video(probe, settings)
    will_change_container = bool(settings["container"]) and \
        settings["container"] != probe["path"].suffix
    if not will_compress and not will_change_container:
        return {"status": "skipped", "outputs": [], "error": "",
                "reason": "nothing to do (bitrate below threshold or no container change)"}

    input_path = probe["path"]
    final_path = build_output_path(
        input_path,
        settings["policy"],
        container=settings["container"],
        output_dir=settings.get("output_dir"),
    )
    temp_path = _temp_for(final_path)
    temp_path.unlink(missing_ok=True)

    desc = "compress H.265" if will_compress else "remux (no re-encode)"
    print(f"  {C.DIM}  {desc}{C.R}")

    cmd = _build_standalone_cmd(input_path, temp_path, settings, probe)
    ok_, err = _run_ffmpeg(cmd, temp_path)
    if not ok_:
        temp_path.unlink(missing_ok=True)
        return {"status": "error", "outputs": [], "error": err, "reason": ""}

    if settings["policy"]["overwrite"]:
        try:
            input_path.unlink(missing_ok=True)
        except OSError as e:
            return {"status": "error", "outputs": [],
                    "error": f"could not delete original: {e}", "reason": ""}

    if final_path.exists() and final_path != input_path:
        final_path.unlink()
    temp_path.replace(final_path)

    return {"status": "ok", "outputs": [final_path], "error": "", "reason": ""}


def _drain_pending_deletes(paths: list) -> tuple:
    """Best-effort delete every path in the list, return (ok_count, failed_count)."""
    ok_n, fail_n = 0, 0
    for p in paths:
        try:
            p.unlink(missing_ok=True)
            ok_n += 1
        except OSError:
            fail_n += 1
    return ok_n, fail_n


def execute_already_dubbed(probes: list, settings: dict) -> dict:
    """Run the full Already-Dubbed batch.

    Two kinds of work units, in order:
      1. Join groups - any base whose segments matched <base>_NN.
      2. Standalone videos - everything not in a group, processed only when
         compress or container change actually applies.
    """
    counters = {
        "processed": 0, "skipped": 0, "errors": 0, "error_log": [],
        "subs_merged": 0, "sub_errors": 0,
    }
    # Rolling cleanup: each iteration drains the *previous* group's source
    # files after the current group's ffmpeg has finished. This caps disk
    # usage at one extra group's worth of leftovers, and gives Windows
    # Defender enough time on the previous output before we ask it to
    # release the segment files.
    prev_pending: list = []
    cleanup_ok = 0
    cleanup_failed = 0
    readable = [p for p in probes if p["ok"]]
    by_path = {p["path"]: p for p in readable}

    groups: dict = {}
    if settings.get("join"):
        groups = detect_split_groups([p["path"] for p in readable])

    grouped_paths = set()
    for items in groups.values():
        for path, _ in items:
            grouped_paths.add(path)

    standalones = [p for p in readable if p["path"] not in grouped_paths]
    do_standalones = bool(settings.get("compress") or settings.get("container"))

    units = []
    for base, items in groups.items():
        seg_paths = [path for path, _ in items]
        primary = by_path.get(seg_paths[0])
        units.append(("join", base, seg_paths, primary))
    if do_standalones:
        for probe in standalones:
            units.append(("standalone", probe["path"].name, None, probe))

    if not units:
        head("Executing")
        warn("Nothing to do - no joinable groups and no standalone op selected.")
        return counters

    head("Executing")
    info(f"{len(groups)} group(s) to join, {len(units) - len(groups)} standalone(s) to process.")
    print()

    total = len(units)
    for i, unit in enumerate(units, 1):
        kind = unit[0]
        if kind == "join":
            _, base, seg_paths, primary = unit
            label = f"join '{base}'  ({len(seg_paths)} segments)"
        else:
            _, _, _, probe = unit
            label = probe["path"].name

        print(
            f"  {C.B}[{i}/{total}] {label}{C.R}    "
            f"{C.DIM}(Press Ctrl+C to Skip this video or to Exit){C.R}"
        )
        try:
            if kind == "join":
                _, base, seg_paths, primary = unit
                if primary is None:
                    res = {"status": "error", "outputs": [],
                           "error": "first segment unreadable", "reason": ""}
                else:
                    res = process_join_group(base, seg_paths, settings, primary)
            else:
                _, _, _, probe = unit
                res = process_standalone_already_dubbed(probe, settings)
        except KeyboardInterrupt:
            if confirm_exit_or_skip():
                warn("Exiting batch by user request.")
                break
            warn("Skipping, continuing with the next item.")
            counters["errors"] += 1
            counters["error_log"].append((label, "interrupted by user"))
            print()
            continue

        if res["status"] == "ok":
            counters["processed"] += 1
            for out in res["outputs"]:
                ok(f"  -> {out.name}")
            sub_out = res.get("sub_output")
            sub_err = res.get("sub_error")
            if sub_out is not None:
                counters["subs_merged"] += 1
                ok(f"  -> {sub_out.name}")
            elif sub_err:
                counters["sub_errors"] += 1
                warn(f"  subtitle: {sub_err}")
                counters["error_log"].append((f"{label} (subtitles)", sub_err))
        elif res["status"] == "skipped":
            counters["skipped"] += 1
            print(f"  {C.DIM}  skipped: {res['reason']}{C.R}")
        else:
            counters["errors"] += 1
            counters["error_log"].append((label, res["error"]))
            bad(f"  failed: {(res['error'] or '')[:200]}")

        # Drain the PREVIOUS group's pending deletes now. The previous
        # group's joined output has had the time it took to encode this one
        # for Defender to scan, so the unlinks should fly.
        if prev_pending:
            ok_now, fail_now = _drain_pending_deletes(prev_pending)
            cleanup_ok += ok_now
            cleanup_failed += fail_now
            print(
                f"  {C.DIM}cleaned up {ok_now} prior source file(s)"
                f"{f' ({fail_now} still locked)' if fail_now else ''}{C.R}"
            )
            prev_pending = []

        # Stage this group's deletes for the next iteration's cleanup.
        if res["status"] == "ok":
            prev_pending = list(res.get("to_delete", []))

        print()

    # The very last group's pending list has nothing after it - drain now.
    if prev_pending:
        ok_now, fail_now = _drain_pending_deletes(prev_pending)
        cleanup_ok += ok_now
        cleanup_failed += fail_now

    if cleanup_ok or cleanup_failed:
        if cleanup_failed:
            warn(
                f"{cleanup_failed} source file(s) could not be deleted "
                "(still locked?). Delete them manually."
            )
        if cleanup_ok:
            ok(f"Removed {cleanup_ok} source file(s).")

    return counters


# ---------------------------------------------------------------------------
# Final report + exit
# ---------------------------------------------------------------------------

def _color_count(n: int, positive_color: str) -> str:
    """Render an integer count with a color (red if errors, green otherwise)."""
    return f"{positive_color}{n}{C.R}"


def print_final_report(
    errors: int, processed: int,
    skipped: int = 0, error_log: Optional[list] = None,
    subs_merged: int = 0, sub_errors: int = 0,
) -> None:
    err_color = C.RED if errors > 0 else C.DIM
    mod_color = C.G if processed > 0 else C.DIM
    print(f"  Error report: {_color_count(errors, err_color)}")
    print(f"  Videos Mod:   {_color_count(processed, mod_color)}")
    if skipped > 0:
        print(f"  Skipped:      {C.DIM}{skipped}{C.R} (no action needed)")
    if subs_merged > 0 or sub_errors > 0:
        sub_color = C.G if subs_merged > 0 else C.DIM
        print(f"  Subs merged:  {sub_color}{subs_merged}{C.R}")
        if sub_errors > 0:
            print(f"  Sub errors:   {C.RED}{sub_errors}{C.R}")
    if error_log:
        print()
        print(f"  {C.RED}Failed videos:{C.R}")
        for name, err in error_log:
            short = err.replace("\n", " ").strip()[:200]
            print(f"    - {name}: {short}")


def mock_final_report() -> None:
    head("Done (no files were modified)")
    print_final_report(errors=0, processed=0)


def _read_one_key() -> str:
    """Read exactly one keystroke without requiring Enter. Returns the
    character (lowercased) or '' if reading raw input fails on this
    platform. Falls back to a normal input() prompt as a last resort."""
    try:
        if os.name == "nt":
            import msvcrt
            raw = msvcrt.getch()
            try:
                return raw.decode("utf-8", errors="ignore").lower()
            except Exception:
                return ""
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1).lower()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        try:
            line = input()
            return line[:1].lower() if line else ""
        except (EOFError, KeyboardInterrupt):
            return ""


def play_done_sound() -> None:
    """Short cheery jingle when a batch finishes. On Windows uses winsound
    to play a six-note flourish through the system audio. Everywhere else
    falls back to the ASCII bell (terminal may or may not honour it)."""
    if os.name == "nt":
        try:
            import winsound
            notes = (
                (659, 120),    # E5
                (784, 120),    # G5
                (1319, 120),   # E6
                (1047, 120),   # C6
                (1175, 120),   # D6
                (1568, 120),   # G6
            )
            for freq, dur in notes:
                winsound.Beep(freq, dur)
            return
        except Exception:
            pass
    sys.stdout.write("\a")
    sys.stdout.flush()


def ask_restart() -> bool:
    """Prompt at the end of a run. Returns True if the user pressed R."""
    print()
    info("Press R to restart, or any other key to exit.")
    return _read_one_key() == "r"


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def main() -> None:
    set_dark_background()
    # Quick Edit Mode is left ON (the Windows default). Selecting text or
    # clicking in the console pauses the visible output until any key is
    # pressed - which used to look like a hang and tempt users into a
    # panicked Ctrl+C. With the Skip/Exit confirmation now in place a
    # stray Ctrl+C no longer kills the batch, so the trade-off has
    # flipped: keeping Quick Edit on is worth it because users get
    # back the click-and-drag selection / copy that Windows users expect.
    print(BANNER)
    check_dependencies()

    while True:
        run_one_pipeline()
        if not ask_restart():
            break


def run_one_pipeline() -> None:
    """One full path / mode / settings / preview / execute / report cycle."""
    folder = ask_path(
        "Where are those MF videos?",
        default=Path(__file__).resolve().parent,
    )

    mode = ask_choice(
        "What are we doing today?",
        {
            "D": "To dub (prepare videos for the dubbing platform)",
            "A": "Already dubbed (post-process dubbed videos)",
        },
        default="D",
    )

    if mode == "D":
        settings = collect_to_dub_settings()
    else:
        settings = collect_already_dubbed_settings()

    proceed, probes, analyses = preview(folder, settings)
    if not proceed:
        warn("Aborted by user.")
        return

    if settings["mode"] == "to_dub":
        counters = execute_to_dub(probes, settings, analyses)
    else:
        counters = execute_already_dubbed(probes, settings)
    head("Done")
    print_final_report(
        errors=counters["errors"],
        processed=counters["processed"],
        skipped=counters["skipped"],
        error_log=counters["error_log"],
        subs_merged=counters.get("subs_merged", 0),
        sub_errors=counters.get("sub_errors", 0),
    )
    play_done_sound()


def _wait_before_exit() -> None:
    # Windows Terminal closes its tab the instant python exits. Block on
    # input so the user can read the final report (or a traceback) instead
    # of watching the window vanish.
    try:
        input("\nPress Enter to close this window...")
    except EOFError:
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n")
        warn("Interrupted by user.")
        sys.exit(130)
    except Exception:
        import traceback
        print("\n")
        bad("Unhandled error - the pipeline crashed:")
        traceback.print_exc()
        _wait_before_exit()
        sys.exit(1)
    else:
        _wait_before_exit()
    finally:
        reset_background()
