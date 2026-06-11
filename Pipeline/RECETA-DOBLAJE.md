# Receta: Doblaje con traducción adaptada (RU → EN/DE/ES/IT)

Pipeline validado el 2026-06-11 con el video "Useful tips from Ruslan Kulabukhov's streams"
(37.5 min, 550 cues): cobertura del guion en el doblaje **99.9%** (los videos fallados con el
flujo viejo daban 26–50%), fidelidad palabra a palabra **98.4%**.

## Flujo completo

1. **Transcribir el video RU** con faster-whisper-xxl generando `srt` + `json` (word timestamps):

   ```
   --model large-v2 --language ru --compute_type int8_float16 --temperature 0
   --beam_size 5 --best_of 1 --task transcribe
   --max_line_width 200 --max_line_count 1 --sentence
   --word_timestamps true --output_format srt json
   ```

   El `json` no respeta `--sentence` (segmentos crudos): no importa, solo se usan las palabras
   sueltas con sus tiempos. El `srt` es idéntico al del flujo viejo (no rompe nada).

2. **Anotar**: `python3 annotate_ru.py video.srt video.json > anotado.txt`
   Produce una línea por cue: `N | ventana | habla Xs | syl~N | texto RU con <pX.X>`
   - `<pX.X>` = pausa real del hablante (gap ≥0.4s entre palabras según el json).
   - `habla` = ventana menos pausas. `syl~N` = presupuesto de sílabas EN (habla × 4.3 síl/s).

3. **Traducir** (Claude — Sonnet 4.6 alcanza, ver abajo) siguiendo las REGLAS.
   Salida: líneas `N|texto traducido`, una por cue, sin saltearse ninguno.

4. **Ensamblar + QA**: `python3 assemble_qa.py fuente.srt salida-EN.srt traducciones1.txt [...]`
   Escribe el srt final (mismos timestamps que el fuente, CRLF) y flaggea cues fuera de
   presupuesto (contador de sílabas calibrado; los cues con números hablados son falsos positivos).

5. **Doblar** en la plataforma con **skip-silent OFF** (las pausas ya viven dentro del texto).

6. **Verificar**: transcribir el video doblado con Whisper y correr
   `python3 compare_dub.py traduccion-EN.srt whisper-del-doblaje.srt`
   Esperable: fidelidad ≥95% (las diferencias son Whisper oyendo mal al TTS: "Kacher"→"catcher"),
   cobertura ≥95%, y la tabla de deriva de sincronía por anclas.

## Reglas de traducción (el corazón de la receta)

1. Inglés hablado coloquial, natural para TTS. Sin rusismos: ну → well; то есть → I mean;
   вот → there / you know; как бы → kind of; допустим → let's say.
2. **Cada `<pX.X>` se preserva como `…`** en la posición semánticamente equivalente de la
   frase traducida. Nunca escribir `<p>` en la salida.
3. **Presupuesto**: para cues con habla ≥1s, apuntar a ±25% de `syl~N`. Comprimir los que
   desbordan; expandir con naturalidad (muletillas, repetición leve) los que quedan cortos.
4. **Registro**: hablante casual y tosco. Mat ruso suavizado a *damn/hell* — nunca más fuerte
   (coherente con las 101 traducciones existentes del catálogo).
5. **Glosario fijo (EN)**: качер = Kacher · катушка = coil · накачка = pumping ·
   заземление = ground/grounding · четверть волны = quarter-wave · противофаза = antiphase ·
   стоячая волна = standing wave · ТВС = flyback · полевик = MOSFET · кондёр = cap ·
   Тесла = Tesla coil · среда = the medium · вакуум = vacuum · Капа/Капанадзе = Kapa/Kapanadze ·
   обратная ЭДС = back-EMF · неонка/`Нюанка` (mishearing) = neon lamp.
6. La transcripción trae errores de Whisper (términos garbled): traducir lo que el hablante
   **probablemente dijo** cuando el contexto lo aclara (остеллограф → oscilloscope, etc.).

## Prompt plantilla para subagente (validado en A/B contra el modelo grande)

> Producción con **Sonnet 4.6**: en el A/B del 2026-06-11 sobre 55 cues con trampas, Sonnet con
> esta receta igualó al modelo grande (pausas 12/12, presupuesto OK, corrigió "Нюанка"→neon lamp).
> El modelo grande se reserva para: cambios de receta, cues flaggeados por el QA, tramos
> ininteligibles. Haiku NO (degrada en ruso coloquial + reparación de mishearings).

```
You are translating Russian transcript cues into English for AI dubbing (TTS).
INPUT: the file <ANOTADO> — read ONLY lines <A> through <B>. Each line:
`N | window | habla Xs | syl~N | Russian text with <pX.X> pause markers`
- habla = speech time (window minus pauses); syl~N = syllable budget (~4.3 syl/s)
- <pX.X> = the speaker paused X.X seconds at that point
OUTPUT: write to <SALIDA>, one line per cue, format `N|English translation`,
covering every cue from <A> to <B> inclusive. No commentary, no skipped cues.
RULES: [reglas 1-6 de arriba, con el glosario completo]
CONTEXT: [una línea sobre el video: quién habla y de qué]
Do not read any other files. Reply "done" plus the line count.
```

Para un video de ~550 cues: 6–10 subagentes en paralelo, ~55–90 cues cada uno, luego
`assemble_qa.py` junta todo y el QA dice si algún bloque necesita retoque.

## Variante local (modelo chico, ej. Gemma 3n E4B)

`traducir_local.py` ejecuta los pasos 2–4 contra un modelo local via API OpenAI-compatible
(Ollama, LM Studio, llama.cpp). Un cue por vez, reanudable (progreso en `.parts.txt`),
con garantias mecanicas para compensar al modelo chico (pausas por construccion, presupuesto
con reintentos, sanitizado, deteccion de cirilico) y `.flags.txt` con los cues a revisar.

```
python3 Pipeline/traducir_local.py --in CARPETA_SRT_JSON --out CARPETA_SALIDA \
    --model gemma3n:e4b --base-url http://localhost:11434/v1 --lang en
```

Validar la calidad del modelo local ANTES de producir en masa: correrlo sobre un video que ya
tenga traduccion de referencia hecha por Claude y comparar con `compare_dub.py` (fidelidad
semantica no aplica ahi, pero sirve `--limit 30` + revision manual de esos 30 contra el gold).

## Estado conocido / pendientes

- **Cobertura y fidelidad: resueltas** con esta receta + skip-silent OFF.
- **Sincronía**: la plataforma no ancla cada cue a su timestamp; la deriva medida en el video
  de prueba osciló entre +18s y −47s a lo largo de 37 min (duración total casi clavada: 36.8
  vs 37.5 min). Tolerable en videos de "cabeza parlante"; molesto en demos con manos. Si hace
  falta sincronía dura: con la traducción (texto→tiempo original) y el Whisper del doblaje
  (texto→tiempo del doblaje) se puede computar el mapa de corrección y re-cortar el audio por
  cue con ffmpeg — pedirle a Claude el "re-sincronizador" mencionando esta receta.
- Glosario DE/ES/IT: derivar de las traducciones existentes en `Ruslan/{DE,ES,IT}` antes de
  producir esos idiomas con subagentes.
