"""Paso 6 de la receta: traduccion adaptada vs Whisper del video doblado.

Uso:  python3 compare_dub.py traduccion-EN.srt whisper-del-doblaje.srt

Reporta:
  - fidelidad global (similitud de texto normalizado + % de palabras verbatim en orden)
  - cobertura (palabras dichas / palabras del guion)
  - primeros cambios reales (tipicamente Whisper oyendo mal al TTS: Kacher -> catcher)
  - deriva de sincronia en anclas (timestamp original vs timestamp en el doblaje)

Interpretacion: fidelidad >=95% y cobertura >=95% = doblaje correcto; las diferencias
restantes son artefactos de transcripcion, no errores de doblaje.
"""
import re, sys, difflib

def parse_srt(path):
    txt = open(path, encoding='utf-8-sig', errors='replace').read().replace('\r\n', '\n')
    cues = []
    for m in re.finditer(r'(\d+)\n(\d\d:\d\d:\d\d[,.]\d\d\d) --> (\d\d:\d\d:\d\d[,.]\d\d\d)\n((?:[^\n]+\n?)+?)(?:\n|$)', txt):
        def s(t):
            t = t.replace(',', '.')
            h, mn, sec = t.split(':')
            return int(h) * 3600 + int(mn) * 60 + float(sec)
        cues.append((s(m.group(2)), s(m.group(3)), ' '.join(m.group(4).strip().split('\n'))))
    return cues

def norm(s):
    return re.sub(r'\s+', ' ', re.sub(r"[^\w\s']", ' ', s.lower())).strip()

def main():
    truth = parse_srt(sys.argv[1])   # traduccion adaptada (texto -> tiempo original)
    dub = parse_srt(sys.argv[2])     # whisper del doblaje (texto -> tiempo del doblaje)

    mn = norm(' '.join(t for _, _, t in truth))
    tn = norm(' '.join(t for _, _, t in dub))
    sim = difflib.SequenceMatcher(None, mn, tn, autojunk=False).ratio() * 100
    mw, tw = mn.split(), tn.split()

    sm = difflib.SequenceMatcher(None, mw, tw, autojunk=False)
    kept = 0
    changes = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            kept += i2 - i1
        elif tag == 'insert':
            changes.append(('+', ' '.join(tw[j1:j2])))
        elif tag == 'delete':
            changes.append(('-', ' '.join(mw[i1:i2])))
        else:
            changes.append(('~', f"{' '.join(mw[i1:i2])} -> {' '.join(tw[j1:j2])}"))

    print(f"fidelidad global: {sim:.1f}%   verbatim en orden: {kept}/{len(mw)} ({kept/len(mw)*100:.1f}%)")
    print(f"cobertura: {len(tw)}/{len(mw)} palabras ({len(tw)/len(mw)*100:.1f}%)")
    print(f"\nprimeros 20 cambios reales:")
    for tag, c in changes[:20]:
        print(f"  {tag} {c[:100]}")

    print(f"\nderiva de sincronia en anclas (timestamp original -> doblaje):")
    print(f"{'cue':>4} {'min':>5} {'drift':>8}  frase")
    dn = [(norm(t), s) for s, e, t in dub]
    step = max(1, len(truth) // 12)
    for i in range(4, len(truth), step):
        mtext = norm(truth[i][2])
        if len(mtext.split()) < 5:
            continue
        best, bs = None, 0
        for ttext, tstart in dn:
            r = difflib.SequenceMatcher(None, mtext, ttext, autojunk=False).ratio()
            if r > bs:
                bs, best = r, tstart
        if bs > 0.6:
            print(f"{i+1:>4} {truth[i][0]/60:>5.1f} {best-truth[i][0]:>+8.1f}s  {truth[i][2][:55]}")

if __name__ == '__main__':
    main()
