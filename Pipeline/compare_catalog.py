"""Compare translated SRTs (Ruslan/{LG}/{N}-{LG}.srt) vs Whisper-from-dubbed SRTs
(Ruslan/{LG}/Subs {LG} TTS/...-{N}-...-{LG}.srt). For each video we measure how
much the TTS-said text matches the translation it was supposed to say.

Per pair we report two numbers:
  sim%   = difflib similarity on normalized text (the headline number)
  len%   = len(TTS) / len(translated) — separates 'TTS shorter because skip-silent'
           from 'TTS said something else'.

Differences are EXPECTED for 'skip silent' segments, number/word formatting,
and Whisper mishearing technical terms. Pairs below the SIM_FLAG threshold get
a detailed side-by-side dump for review.
"""
import re, sys, difflib
from pathlib import Path

ROOT = Path('/home/user/Sta-RU/Ruslan')
LANGS = sys.argv[1:] or ['DE', 'EN']
SIM_FLAG = 70.0          # flag pairs below this for the detail report
DETAIL_TOPN = 12         # also dump the worst N per language regardless of threshold

def srt_text(content):
    """Concatenate all cue content into one string. SRT parser-light: skip the
    index line and the timestamp line, keep everything else until a blank."""
    lines = content.replace('\r\n', '\n').split('\n')
    out, i = [], 0
    while i < len(lines):
        ln = lines[i].strip()
        if re.fullmatch(r'\d+', ln):     # cue index
            i += 1
            if i < len(lines) and '-->' in lines[i]:
                i += 1
            buf = []
            while i < len(lines) and lines[i].strip():
                buf.append(lines[i].strip())
                i += 1
            if buf:
                out.append(' '.join(buf))
        i += 1
    return ' '.join(out)

def normalize(s):
    s = s.lower()
    s = re.sub(r"[‘’´`]", "'", s)
    s = re.sub(r'[^\w\s\']', ' ', s, flags=re.UNICODE)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def load_srt(path):
    try:
        return path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        return path.read_text(encoding='latin-1')

def parse_n(name):
    m = re.search(r' - (\d+) - ', name)
    return int(m.group(1)) if m else None

def collect_translated(lg):
    out = {}
    for p in (ROOT / lg).glob(f'*-{lg}.srt'):
        m = re.match(r'(\d+)-', p.name)
        if m:
            out[int(m.group(1))] = p
    return out

def collect_tts(lg):
    out = {}
    folder = ROOT / lg / f'Subs {lg} TTS'
    if not folder.is_dir():
        return out
    for p in folder.glob('*.srt'):
        n = parse_n(p.name)
        if n is not None:
            out.setdefault(n, p)   # first one wins if duplicates
    return out

def compare(transl_text, tts_text):
    """Returns (sim_pct, len_pct)."""
    if not transl_text:
        return 0.0, 0.0
    sim = difflib.SequenceMatcher(None, transl_text, tts_text, autojunk=False).ratio() * 100
    len_pct = (len(tts_text) / len(transl_text)) * 100 if transl_text else 0.0
    return sim, len_pct

def short_diff(transl_norm, tts_norm, width=110):
    """Compact unified-style diff of normalized text, line-wrapped."""
    def wrap(s):
        return [s[i:i+width] for i in range(0, len(s), width)] or ['']
    diff = difflib.unified_diff(wrap(transl_norm), wrap(tts_norm),
                                fromfile='translated', tofile='TTS',
                                lineterm='', n=1)
    return '\n'.join(diff)

def main():
    detail_path = Path('/tmp/compare_details.txt')
    summary_lines = []
    detail_out = []
    grand_low = []

    for lg in LANGS:
        transl = collect_translated(lg)
        tts = collect_tts(lg)
        all_ns = sorted(set(transl) | set(tts))
        if not all_ns:
            print(f'[{lg}] no files found in {ROOT/lg}')
            continue

        rows = []
        only_transl = []
        only_tts = []
        for n in all_ns:
            tp = transl.get(n); xp = tts.get(n)
            if not tp:
                only_tts.append(n); continue
            if not xp:
                only_transl.append(n); continue
            t_raw = srt_text(load_srt(tp))
            x_raw = srt_text(load_srt(xp))
            t_norm = normalize(t_raw); x_norm = normalize(x_raw)
            sim, lenp = compare(t_norm, x_norm)
            rows.append((n, sim, lenp, len(t_norm), len(x_norm), tp, xp, t_norm, x_norm))

        rows.sort(key=lambda r: r[1])          # worst similarity first

        sims = [r[1] for r in rows]
        avg = sum(sims)/len(sims) if sims else 0
        med = sorted(sims)[len(sims)//2] if sims else 0
        flagged = [r for r in rows if r[1] < SIM_FLAG]

        print(f"\n{'='*78}\n[{lg}]  pairs: {len(rows)}  "
              f"avg sim: {avg:.1f}%  median: {med:.1f}%  "
              f"flagged (<{SIM_FLAG:.0f}%): {len(flagged)}\n"
              f"{'='*78}")
        if only_transl:
            print(f'  translated but no TTS: {only_transl}')
        if only_tts:
            print(f'  TTS but no translated: {only_tts}')
        print(f"\n  {'N#':>4}  {'sim%':>6}  {'len%':>6}  {'tlen':>6}  {'xlen':>6}")
        for n, sim, lenp, tl, xl, *_ in rows:
            flag = '  <-- review' if sim < SIM_FLAG else ''
            print(f"  {n:>4}  {sim:>6.1f}  {lenp:>6.1f}  {tl:>6d}  {xl:>6d}{flag}")

        summary_lines.append(
            f'{lg}: {len(rows)} pairs, avg {avg:.1f}%, median {med:.1f}%, flagged {len(flagged)}'
        )

        # Detail: flagged + bottom-N (union, dedup) so we always inspect the worst
        worst_ns = {r[0] for r in rows[:DETAIL_TOPN]} | {r[0] for r in flagged}
        for n, sim, lenp, tl, xl, tp, xp, t_norm, x_norm in rows:
            if n not in worst_ns: continue
            detail_out.append(
                f'\n{"#"*78}\n# [{lg}] N#{n}  sim={sim:.1f}%  len={lenp:.1f}%  '
                f'(translated {tl} chars vs TTS {xl} chars)\n'
                f'#   translated: {tp.name}\n#   TTS:        {xp.name}\n{"#"*78}\n'
                f'-- TRANSLATED (normalized) --\n{t_norm[:2400]}'
                f'{" ..." if len(t_norm)>2400 else ""}\n\n'
                f'-- TTS (normalized) --\n{x_norm[:2400]}'
                f'{" ..." if len(x_norm)>2400 else ""}\n\n'
                f'-- short diff --\n{short_diff(t_norm[:2400], x_norm[:2400])}\n'
            )
            if sim < SIM_FLAG:
                grand_low.append((lg, n, sim, lenp))

    print('\n' + '='*78 + '\nSUMMARY\n' + '='*78)
    for line in summary_lines:
        print('  ' + line)
    if grand_low:
        print(f'\n  Flagged pairs ({len(grand_low)} total, sim<{SIM_FLAG:.0f}%):')
        for lg, n, sim, lenp in sorted(grand_low, key=lambda x: x[2]):
            print(f'    {lg} N#{n}: sim={sim:.1f}% len={lenp:.1f}%')
    else:
        print(f'\n  No pairs below {SIM_FLAG:.0f}% similarity.')

    detail_path.write_text(''.join(detail_out), encoding='utf-8')
    print(f'\nDetail dump (worst per language + flagged): {detail_path}'
          f' ({detail_path.stat().st_size//1024} KB)')

if __name__ == '__main__':
    main()
