"""Paso 4 de la receta: junta las traducciones (lineas `N|texto`) + srt fuente
-> srt final traducido (mismos timestamps, CRLF) + QA de presupuesto de silabas.

Uso:  python3 assemble_qa.py fuente.srt salida-EN.srt traducciones1.txt [traducciones2.txt ...]

QA: flaggea cues cuya duracion estimada de habla EN queda fuera de [0.50, 1.45] x el tiempo
de habla real del original. Contador calibrado (e muda, -ed, -es); los cues con numeros
hablados ("TC-44") son falsos positivos conocidos: el contador no cuenta digitos.
"""
import re, sys

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
        # habla real = ventana menos pausas; aproximamos con la ventana
        # (el anotado tiene el dato exacto, pero el srt fuente alcanza para flaggear)
        if win < 1.0:
            continue
        r = (syl(trans[n]) / RATE) / win
        if r > OVER or r < UNDER:
            flagged += 1
            print(f"  QA cue {n}: ratio {r:.2f} ({'desborde' if r > OVER else 'corto'}) | {trans[n][:80]}")
    print(f"QA: {flagged} cues flaggeados (revisar; numeros hablados = falso positivo)")

if __name__ == '__main__':
    main()
