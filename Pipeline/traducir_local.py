"""Traduccion local de subtitulos para doblaje (receta Pipeline/RECETA-DOBLAJE.md)
usando un modelo local via API OpenAI-compatible (Ollama, LM Studio, llama.cpp...).

Pensado para modelos chicos (ej. Gemma 3n E4B): traduce UN cue por vez, con
contexto rodante, y compensa la fragilidad del modelo con garantias mecanicas:
  - sanitizado de salida (comillas, prefijos, lineas extra)
  - deteccion de cirilico -> reintento -> traduccion por fragmentos
  - pausas: los marcadores <pX.X> se preservan; si el modelo los pierde, se
    traduce fragmento por fragmento y las pausas quedan bien POR CONSTRUCCION
  - presupuesto de silabas: reintentos "mas corto/mas largo", se queda con el
    mejor intento; cues con numeros no se fuerzan (el contador no lee digitos)
  - reanudable: el progreso se guarda cue a cue en <video>.parts.txt; si se
    corta a mitad de la noche, re-ejecutar continua donde quedo
  - cues problematicos quedan listados en <video>.flags.txt para revision

Uso:
  python3 traducir_local.py --in CARPETA --out CARPETA_SALIDA \
      [--model gemma3n:e4b] [--base-url http://localhost:11434/v1] \
      [--lang en] [--limit N] [--force]

Busca en --in cada par <stem>.srt + <stem>.json (word timestamps de
faster-whisper-xxl). Salida: <stem>-EN.srt (si el stem termina en -RU, lo
reemplaza). Sin dependencias: solo libreria estandar (Python 3.8+).
"""
import argparse, json, os, re, sys, time, urllib.request

# ---------------- configuracion por idioma ----------------
LANGS = {
    "en": {
        "rate": 4.3,        # silabas/seg comodas para TTS (presupuesto)
        "rate_qa": 4.6,     # silabas/seg para estimar duracion hablada (QA)
        "system": (
            "You are a professional Russian-to-English translator for AI dubbing (text-to-speech).\n"
            "Rules:\n"
            "- Colloquial spoken English, natural when read aloud. No Russianisms: "
            "ну = well; то есть = I mean; вот = there / you know; как бы = kind of; допустим = let's say.\n"
            "- Keep every <pX.X> marker EXACTLY as written, at the semantically equivalent position. "
            "Never translate, move to the end, or delete them.\n"
            "- The speaker is a casual, rough-spoken Russian DIY electronics YouTuber. "
            "Soften profanity to damn/hell level only.\n"
            "- Glossary, always: качер = Kacher; катушка = coil; накачка = pumping; заземление = ground; "
            "четверть волны = quarter-wave; противофаза = antiphase; стоячая волна = standing wave; "
            "ТВС = flyback; полевик = MOSFET; кондёр = cap; Тесла = Tesla coil; среда = the medium; "
            "вакуум = vacuum; Капа/Капанадзе = Kapa/Kapanadze; обратная ЭДС = back-EMF; неонка = neon lamp.\n"
            "- The transcript may contain speech-recognition errors; translate what the speaker most likely said.\n"
            "- Reply with ONLY the translation, one single line, no quotes, no notes, no explanations."
        ),
    },
    # "es" / "de" / "it": agregar bloque cuando EN este confirmado
}

PAUSE = 0.4
P_RE = re.compile(r'<p[\d.]+>')

# ---------------- srt / json / anotacion ----------------
def parse_srt(path):
    txt = open(path, encoding='utf-8-sig', errors='replace').read().replace('\r\n', '\n')
    cues = []
    for m in re.finditer(r'(\d+)\n(\d\d:\d\d:\d\d[,.]\d\d\d --> \d\d:\d\d:\d\d[,.]\d\d\d)\n((?:[^\n]+\n?)+?)(?:\n|$)', txt):
        ts = m.group(2)
        def s(t):
            h, mn, sec = t.replace(',', '.').split(':')
            return int(h) * 3600 + int(mn) * 60 + float(sec)
        a, b = ts.split(' --> ')
        cues.append({'ts': ts, 'start': s(a), 'end': s(b),
                     'text': ' '.join(m.group(3).strip().split('\n'))})
    return cues

def annotate(cues, json_path):
    d = json.load(open(json_path, encoding='utf-8'))
    words = [w for seg in d['segments'] for w in seg.get('words', [])]
    for c in cues:
        cw = [w for w in words if w['start'] >= c['start'] - 0.05 and w['start'] < c['end']]
        win = c['end'] - c['start']
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
            c['marked'] = ' '.join(parts)
        else:
            c['marked'] = c['text']
        c['speech'] = max(win - pause_t, 0.3)

def syl_word_en(w):
    w = w.lower()
    g = len(re.findall(r'[aeiouy]+', w))
    if g > 1:
        if re.search(r'[^aeiouy]e$', w): g -= 1
        elif re.search(r'[^aeiouytd]ed$', w): g -= 1
        elif re.search(r'[^aeiouycsxz]es$', w): g -= 1
    return max(1, g)

def syl_en(t):
    return sum(syl_word_en(w) for w in re.findall(r"[a-z']+", t.lower()))

# ---------------- cliente API local ----------------
class LocalLLM:
    def __init__(self, base_url, model, timeout):
        self.url = base_url.rstrip('/') + '/chat/completions'
        self.model = model
        self.timeout = timeout

    def chat(self, system, user, max_tokens=220):
        body = json.dumps({
            "model": self.model, "stream": False,
            "temperature": 0.2, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }).encode('utf-8')
        delays = [0, 10, 30, 60, 120]
        for i, d in enumerate(delays):
            if d: time.sleep(d)
            try:
                req = urllib.request.Request(self.url, data=body,
                                             headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read().decode('utf-8'))
                return out['choices'][0]['message']['content']
            except Exception as e:
                if i == len(delays) - 1:
                    raise SystemExit(f"ERROR: la API local no responde tras {len(delays)} intentos: {e}")
                print(f"  (API fallo: {e} — reintento en {delays[i+1]}s)", flush=True)

# ---------------- traduccion de un cue ----------------
CYR = re.compile(r'[а-яА-ЯёЁ]')
LABEL = re.compile(r'^\s*(translation|english|en|respuesta|output)\s*[:\-]\s*', re.I)

def sanitize(raw):
    lines = [l.strip() for l in raw.strip().split('\n') if l.strip()]
    lines = [l for l in lines if not re.match(r'^```', l)]
    if not lines:
        return ''
    t = lines[0]
    if LABEL.match(t) and len(lines) > 1 and len(LABEL.sub('', t)) < 3:
        t = lines[1]
    t = LABEL.sub('', t)
    t = re.sub(r'^\d+\s*\|\s*', '', t)
    t = t.strip().strip('"').strip('«»').strip("'").strip()
    return re.sub(r'\s+', ' ', t)

def ctx_block(context):
    if not context:
        return ''
    out = "Context (previous lines, already translated):\n"
    for ru, en in context:
        out += f"RU: {ru}\nEN: {en}\n"
    return out + "\n"

def translate_fragments(llm, cfg, marked, context):
    full = re.sub(r'\s+', ' ', P_RE.sub(' ', marked)).strip()
    frags = [f.strip() for f in P_RE.split(marked) if f.strip()]
    done = []
    for fr in frags:
        user = (ctx_block(context) +
                "This Russian sentence is split by speech pauses. Translate ONLY the fragment "
                "so the fragments read naturally in sequence.\n"
                f"Full sentence: {full}\n" +
                (f"Fragments already translated: {' … '.join(done)}\n" if done else '') +
                f"Fragment: {fr}\nOnly the fragment translation:")
        t = sanitize(llm.chat(cfg['system'], user, max_tokens=120))
        if CYR.search(t) or not t:
            t = sanitize(llm.chat(cfg['system'], user + "\nYour answer must contain NO Russian letters.", max_tokens=120))
        done.append(t if t else '...')
    return ' … '.join(done)

def translate_cue(llm, cfg, cue, context):
    flags = []
    marked = cue['marked']
    src_p = len(P_RE.findall(marked))
    budget = int(cue['speech'] * cfg['rate'])

    user = (ctx_block(context) +
            f"Translate this line (aim for about {budget} spoken syllables):\n"
            f"RU: {marked}\nEN:")
    out = sanitize(llm.chat(cfg['system'], user))

    if not out or CYR.search(out):
        out2 = sanitize(llm.chat(cfg['system'], user + "\nYour answer must contain NO Russian letters."))
        out = out2 if out2 and not CYR.search(out2) else ''
    # pausas
    if src_p > 0:
        if out and len(P_RE.findall(out)) == src_p:
            out = re.sub(r'\s*<p[\d.]+>\s*', ' … ', out).strip()
        elif out and out.count('…') + out.count('...') >= src_p:
            out = P_RE.sub(' ', out)
            out = re.sub(r'\s+', ' ', out).strip()
        else:
            out = translate_fragments(llm, cfg, marked, context)
            flags.append('pausas por fragmentos')
    else:
        out = P_RE.sub(' ', out)
        out = re.sub(r'\s+', ' ', out).strip()
    if not out:
        return '…', flags + ['SIN TRADUCCION — revisar']
    if CYR.search(out):
        flags.append('CIRILICO en salida — revisar')

    # presupuesto (solo cues sin digitos, habla >= 1s)
    if cue['speech'] >= 1.0 and not re.search(r'\d', marked) and not re.search(r'\d', out):
        def ratio(t):
            return (syl_en(t) / cfg['rate_qa']) / cue['speech']
        attempts = [(abs(ratio(out) - 1.0), out)]
        for _ in range(2):
            cur = min(attempts)[1]
            r = ratio(cur)
            if 0.50 <= r <= 1.40:
                break
            ask = ("Rewrite this English line SHORTER (about {b} spoken syllables), same meaning"
                   if r > 1.40 else
                   "Rewrite this English line a bit LONGER (about {b} spoken syllables), natural fillers allowed, same meaning")
            user2 = (f"{ask.format(b=budget)}, keep any … exactly where they are.\n"
                     f"Russian original: {marked}\nLine: {cur}\nOnly the rewritten line:")
            t = sanitize(llm.chat(cfg['system'], user2))
            if t and not CYR.search(t):
                if src_p > 0 and t.count('…') + t.count('...') < src_p:
                    pass  # perdio pausas: no lo aceptamos
                else:
                    attempts.append((abs(ratio(t) - 1.0), t))
        out = min(attempts)[1]
        r = ratio(out)
        if r > 1.45 or r < 0.50:
            flags.append(f'presupuesto ratio {r:.2f}')
    return out, flags

# ---------------- proceso por video ----------------
def out_name(stem, lang):
    base = re.sub(r'-RU$', '', stem)
    return f"{base}-{lang.upper()}.srt"

def process_video(llm, cfg, lang, srt_path, json_path, out_dir, limit):
    stem = os.path.splitext(os.path.basename(srt_path))[0]
    dst = os.path.join(out_dir, out_name(stem, lang))
    parts_path = os.path.join(out_dir, stem + '.parts.txt')
    flags_path = os.path.join(out_dir, stem + '.flags.txt')

    cues = parse_srt(srt_path)
    annotate(cues, json_path)
    total = len(cues)

    done = {}
    if os.path.exists(parts_path):
        for line in open(parts_path, encoding='utf-8'):
            line = line.rstrip('\n')
            if '|' in line:
                n, t = line.split('|', 1)
                done[int(n)] = t
    print(f"\n=== {stem}: {total} cues, {len(done)} ya hechos ===", flush=True)

    todo = [i for i in range(total) if (i + 1) not in done]
    if limit:
        todo = todo[:max(0, limit - len(done))]

    context = []
    t0, n_done_now = time.time(), 0
    with open(parts_path, 'a', encoding='utf-8') as pf, \
         open(flags_path, 'a', encoding='utf-8') as ff:
        for i in range(total):
            n = i + 1
            if n in done:
                context = (context + [(P_RE.sub(' ', cues[i]['marked']).strip(), done[n])])[-2:]
                continue
            if limit and n_done_now >= len(todo):
                break
            text, flags = translate_cue(llm, cfg, cues[i], context)
            pf.write(f"{n}|{text}\n"); pf.flush()
            for fl in flags:
                ff.write(f"cue {n}: {fl} | {text[:80]}\n"); ff.flush()
            done[n] = text
            context = (context + [(P_RE.sub(' ', cues[i]['marked']).strip(), text)])[-2:]
            n_done_now += 1
            if n_done_now % 25 == 0:
                rate = (time.time() - t0) / n_done_now
                eta = rate * (total - len(done)) / 60
                print(f"  {len(done)}/{total} cues  ({rate:.1f}s/cue, ETA {eta:.0f} min)", flush=True)

    if len(done) == total:
        out = [f"{n}\n{cues[n-1]['ts']}\n{done[n]}\n" for n in range(1, total + 1)]
        open(dst, 'w', encoding='utf-8', newline='\r\n').write('\n'.join(out) + '\n')
        nf = sum(1 for _ in open(flags_path, encoding='utf-8')) if os.path.exists(flags_path) else 0
        print(f"  LISTO: {dst}  ({total} cues, {nf} flags en {os.path.basename(flags_path)})", flush=True)
    else:
        print(f"  parcial: {len(done)}/{total} — re-ejecutar para continuar", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='indir', required=True)
    ap.add_argument('--out', dest='outdir', required=True)
    ap.add_argument('--base-url', default='http://localhost:11434/v1')
    ap.add_argument('--model', default='gemma3n:e4b')
    ap.add_argument('--lang', default='en', choices=list(LANGS))
    ap.add_argument('--limit', type=int, default=0, help='traducir solo N cues por video (prueba)')
    ap.add_argument('--timeout', type=int, default=300)
    ap.add_argument('--force', action='store_true', help='rehacer videos ya completados')
    a = ap.parse_args()

    cfg = LANGS[a.lang]
    os.makedirs(a.outdir, exist_ok=True)
    llm = LocalLLM(a.base_url, a.model, a.timeout)

    pairs = []
    for f in sorted(os.listdir(a.indir)):
        if f.lower().endswith('.srt'):
            stem = os.path.splitext(f)[0]
            j = os.path.join(a.indir, stem + '.json')
            if os.path.exists(j):
                pairs.append((os.path.join(a.indir, f), j))
    if not pairs:
        sys.exit(f"no hay pares .srt+.json en {a.indir}")
    print(f"{len(pairs)} videos | modelo {a.model} @ {a.base_url} | idioma {a.lang}")

    for srt_path, json_path in pairs:
        stem = os.path.splitext(os.path.basename(srt_path))[0]
        dst = os.path.join(a.outdir, out_name(stem, a.lang))
        if os.path.exists(dst) and not a.force:
            print(f"\n=== {stem}: ya completado ({dst}) — salteo ===")
            continue
        process_video(llm, cfg, a.lang, srt_path, json_path, a.outdir, a.limit)

if __name__ == '__main__':
    main()
