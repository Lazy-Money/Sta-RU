"""Standalone XTTS-v2 segment generator.

Strips the pipeline down to its core: load a voice reference WAV, read
an SRT, and emit one .wav per subtitle line — nothing else. Use it to
A/B specific lines, compare voice refs, or audit which texts XTTS
robotizes vs. handles cleanly.

Usage:
    python test_xtts.py VOICE_REF.wav SUBTITLES.srt [OUT_DIR] [--lang en]

Outputs:
    OUT_DIR/seg_0001.wav    (with sidecar seg_0001.txt holding the line)
    OUT_DIR/seg_0002.wav
    ...
    OUT_DIR/index.tsv       (n<TAB>start<TAB>end<TAB>chars<TAB>text)

Defaults: OUT_DIR = ./xtts_test_out, --lang = en
"""

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Generate one XTTS-v2 WAV per SRT line.")
    p.add_argument("voice_ref", type=Path, help="Speaker reference WAV (clean, ~6-15s).")
    p.add_argument("srt", type=Path, help="Subtitle file.")
    p.add_argument("out_dir", type=Path, nargs="?", default=Path("xtts_test_out"),
                   help="Output folder (default: ./xtts_test_out).")
    p.add_argument("--lang", default="en", help="XTTS language code (default: en).")
    p.add_argument("--temperature", type=float, default=0.75,
                   help="XTTS sampling temperature (default: 0.75).")
    args = p.parse_args()

    if not args.voice_ref.is_file():
        print(f"ERR: voice ref not found: {args.voice_ref}", file=sys.stderr)
        return 1
    if not args.srt.is_file():
        print(f"ERR: srt not found: {args.srt}", file=sys.stderr)
        return 1

    import srt
    import torch
    from TTS.api import TTS

    with open(args.srt, encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))
    print(f"Loaded {len(subs)} subtitle entries from {args.srt.name}")

    os.environ["COQUI_TOS_AGREED"] = "1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading XTTS-v2 on {device}...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print(f"Model ready. Voice ref: {args.voice_ref.name}\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index_rows = ["n\tstart\tend\tchars\ttext"]
    n_done = 0
    n_skipped = 0
    n_failed = 0

    for sub in subs:
        n = sub.index
        text = sub.content.strip().replace("\n", " ")
        start = sub.start.total_seconds()
        end = sub.end.total_seconds()
        if not text:
            n_skipped += 1
            continue
        seg_path = args.out_dir / f"seg_{n:04d}.wav"
        txt_path = args.out_dir / f"seg_{n:04d}.txt"
        print(f"[{n:>4}] {len(text):>3} chars  {start:6.2f}-{end:6.2f}  {text[:70]}",
              flush=True)
        try:
            tts.tts_to_file(
                text=text,
                file_path=str(seg_path),
                speaker_wav=str(args.voice_ref),
                language=args.lang,
                split_sentences=False,
                temperature=args.temperature,
            )
            txt_path.write_text(text, encoding="utf-8")
            index_rows.append(f"{n}\t{start:.3f}\t{end:.3f}\t{len(text)}\t{text}")
            n_done += 1
        except Exception as e:
            print(f"       [FAIL] {e}", file=sys.stderr)
            n_failed += 1

    (args.out_dir / "index.tsv").write_text("\n".join(index_rows) + "\n", encoding="utf-8")
    print(f"\nDone — {n_done} ok, {n_skipped} empty, {n_failed} failed")
    print(f"Output: {args.out_dir.resolve()}")
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
