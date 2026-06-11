"""Paso 2 de la receta: srt + json (word timestamps) -> anotado para traducir.

Uso:  python3 annotate_ru.py video.srt video.json > anotado.txt

Salida, una linea por cue:
  N | <ventana>s | habla <X>s | syl~<N> | texto RU con marcadores <pX.X>

<pX.X> = pausa real del hablante (gap >= PAUSE entre palabras consecutivas del json).
habla   = ventana menos pausas internas.
syl~N   = presupuesto de silabas EN habladas (habla * 4.3 sil/s).
"""
import json, re, sys

PAUSE = 0.4
EN_RATE = 4.3

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

def main():
    srt_path, json_path = sys.argv[1], sys.argv[2]
    cues = parse_srt(srt_path)
    d = json.load(open(json_path, encoding='utf-8'))
    words = [w for seg in d['segments'] for w in seg.get('words', [])]

    for idx, (cs, ce, ctext) in enumerate(cues, 1):
        cw = [w for w in words if w['start'] >= cs - 0.05 and w['start'] < ce]
        win = ce - cs
        pause_t = 0.0
        if len(cw) >= 2:
            parts = []
            for i, w in enumerate(cw):
                if i > 0:
                    gap = w['start'] - cw[i - 1]['end']
                    if gap >= PAUSE:
                        parts.append(f"<p{gap:.1f}>")
                        pause_t += gap
                parts.append(w['word'].strip())
            marked = ' '.join(parts)
        else:
            marked = ctext
        speech = max(win - pause_t, 0.3)
        syl = int(speech * EN_RATE)
        print(f"{idx} | {win:.1f}s | habla {speech:.1f}s | syl~{syl} | {marked}")

if __name__ == '__main__':
    main()
