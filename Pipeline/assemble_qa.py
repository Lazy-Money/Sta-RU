"""Paso 4 de la receta: junta las traducciones (lineas `N|texto`) + srt fuente
-> srt final traducido (mismos timestamps, CRLF) + QA de presupuesto de silabas.

Uso:  python3 assemble_qa.py fuente.srt salida-EN.srt [anotado.txt] trads1.txt [trads2.txt ...]

Si se pasa el anotado (salida de annotate_ru.py, se detecta solo), el QA usa el tiempo de
habla real (ventana menos pausas); si no, aproxima con la ventana y sobre-flaggea cues con
pausas largas. Flaggea fuera de [0.50, 1.45]. Contador calibrado (e muda, -ed, -es); los
cues con numeros hablados ("TC-44") son falsos positivos conocidos: no cuenta digitos.
"""
import re, sys

ANN_RE = re.compile(r'(\d+) \| [\d.]+s \| habla ([\d.]+)s \| syl~\d+ \|')

RATE = 4.6
OVER, UNDER = 1.45, 0.50

def parse_srt(path):
    txt = open(path, encoding='utf-8-sig', errors='replace').read().replace('\r\n', '\n')
    cues = []
    for m in re.finditer(r'(\d+)\n(\d\d:\d\d:\d\d[,.]\d\d\d --> \d\d:\d\d:\d\d[,.]\d\d\d)\n((?:[^\n]+\n?)+?)(?:\n|$)', txt):
        cues.append((int(m.group(1)), m.group(2), ' '.join(m.group(3).strip().split('\n'))))
    return cues

def ts2s(ts):
    h, mn, sec = ts.replace(',', '.').split(':')
    return int(h) * 3600 + int(mn) * 60 + float(sec)

def syl_word(w):
    w = w.lower()
    g = len(re.findall(r'[aeiouy]+', w))
    if g > 1:
        if re.search(r'[^aeiouy]e$', w): g -= 1
        elif re.search(r'[^aeiouytd]ed$', w): g -= 1
        elif re.search(r'[^aeiouycsxz]es$', w): g -= 1
    return max(1, g)

def syl(t):
    return sum(syl_word(w) for w in re.findall(r"[a-z']+", t.lower()))

def main():
    src_path, dst_path, *chunk_paths = sys.argv[1:]
    cues = parse_srt(src_path)

    # deteccion del anotado opcional: archivo cuya primera linea matchea el formato
    habla = {}
    rest = []
    for p in chunk_paths:
        first = open(p, encoding='utf-8').readline()
        if ANN_RE.match(first):
            for line in open(p, encoding='utf-8'):
                m = ANN_RE.match(line)
                if m:
                    habla[int(m.group(1))] = float(m.group(2))
        else:
            rest.append(p)
    chunk_paths = rest

    trans = {}
    for p in chunk_paths:
        for line in open(p, encoding='utf-8'):
            line = line.rstrip('\n')
            if not line.strip():
                continue
            n, t = line.split('|', 1)
            n = int(n)
            if n in trans and trans[n] != t.strip():
                print(f"AVISO: cue {n} duplicado con texto distinto (gana el ultimo: {p})")
            trans[n] = t.strip()

    missing = [n for n, _, _ in cues if n not in trans]
    if missing:
        sys.exit(f"ERROR: faltan traducciones para los cues {missing}")
    empty = [n for n, t in trans.items() if not t]
    if empty:
        sys.exit(f"ERROR: cues vacios {empty}")

    out = [f"{n}\n{ts}\n{trans[n]}\n" for n, ts, _ in cues]
    open(dst_path, 'w', encoding='utf-8', newline='\r\n').write('\n'.join(out) + '\n')
    print(f"OK: {dst_path} ({len(cues)} cues)")

    flagged = 0
    for n, ts, _ in cues:
        a, b = ts.split(' --> ')
        win = ts2s(b) - ts2s(a)
        speech = habla.get(n, win)
        if speech < 1.0:
            continue
        r = (syl(trans[n]) / RATE) / speech
        if r > OVER or r < UNDER:
            flagged += 1
            print(f"  QA cue {n}: ratio {r:.2f} ({'desborde' if r > OVER else 'corto'}) | {trans[n][:80]}")
    print(f"QA: {flagged} cues flaggeados (revisar; numeros hablados = falso positivo)")

if __name__ == '__main__':
    main()
