## Sta-RU

Subtítulos del canal de Ruslan (electrónica DIY, generadores, "free energy",
bobina Kacher) en RU, EN, DE, ES, IT — más infraestructura de transcripción
y doblaje en `colab/`. Mantenedor en Argentina; idioma de la conversación: ES.

### Estructura

- `Ruslan/{RU,EN,DE,ES,IT}/<N>-<LANG>.srt` — los subs numerados 1..N son la
  serie "estable" (EN es la más completa, ~365 archivos).
- `Ruslan/<LANG>/Missing/<título>-<LANG>.srt` — lotes nuevos que todavía no
  entraron a la numeración. Se traducen con el mismo workflow que abajo y se
  espejan en EN/DE/ES/IT con el mismo nombre.
- `Ruslan/Test/` — material de prueba, no producción.
- `colab/Sta_RU_*.ipynb` — notebooks Colab (transcribir RU desde YouTube,
  doblar con TTS, comparar Demucs, etc.).
- `colab/{batch_dub,batch_dub_edge,video_pipeline,split_pipeline}.py` — pipeline
  de doblaje.

### Workflow para traducir SRT nuevos (IMPORTANTE — preguntármelo es señal de que olvidaste leer este archivo)

Cuando aparecen SRT nuevos en `Ruslan/RU/...` y hay que traducirlos:

1. **RU → EN** delegado a un subagente Sonnet 4.6
   (`Agent` con `subagent_type: general-purpose`, `model: sonnet`).
   El ruso de Ruslan es coloquial-técnico (Kacher, генератор, лавинный блок,
   "ну да", "так"); Sonnet 4.6 lo maneja mejor que traductores automáticos.
2. **Yo (Claude principal) reviso los EN** archivo por archivo:
   - Sin **rusismos**: nada de "Yes-yes", "Well-well", calcos de orden de
     palabras o partículas (`ну`, `вот`, `так`) traducidas literalmente.
   - **Tono coloquial natural** en inglés (Ruslan habla informal a cámara).
   - **Términos técnicos rusos establecidos** se conservan transliterados:
     `Kacher` (no "Kacher coil"), `Tesla coil` solo si el RU dice `катушка Тесла`.
   - **Nombres propios** intactos: Oleg, Roma, Ruslan, Kulobukhov, etc.
   - **No re-segmentar**: índices y timestamps idénticos al RU, una sola línea
     de texto por bloque cuando entra.
3. **EN → DE/ES/IT** delegado a subagentes Sonnet 4.6 — uno por idioma, en
   **paralelo** (una sola tool-call con tres `Agent` simultáneos). Pivot
   desde el EN ya revisado, **no** desde el RU.

### Convención de SRT
- UTF-8 sin BOM. Formato SRT estándar (índice, `HH:MM:SS,mmm --> HH:MM:SS,mmm`,
  texto, línea en blanco).
- Timestamps y numeración **idénticos** entre los 5 idiomas.
- No se "corrige" el RU si Whisper lo dejó raro — la fuente de verdad es el RU
  transcripto; si está mal, se **re-transcribe**, no se parchea en EN.

### Notebook de transcripción `colab/Sta_RU_YouTube_Transcribe_RU.ipynb`
- Solo 2 celdas: setup+links / procesar todo.
- `WhisperModel("large-v2")`, `language="ru"`, `task="transcribe"`,
  `temperature=0`, `beam_size=5`, `best_of=1`, `vad_filter=True`,
  `condition_on_previous_text=True` (mismos parámetros que el script de
  PowerShell `faster-whisper-xxl` que usa Ruslan local).
- Dedupe por video id; si dos videos distintos chocan en nombre, agrega
  `[<youtube_id>]` al stem.
- El ZIP de salida empaqueta solo los SRT de la corrida (no glob del dir).

### Git
- Branch activa esta sesión: `claude/compassionate-knuth-M2llL`.
- Commits descriptivos, prefijo por área cuando aplique:
  `colab(youtube-transcribe): ...`, `Translate Ruslan subtitles to ES/DE/IT (batch)`,
  `Add dubbing-adapted EN translations for ...`.
- Push siempre con `-u origin <branch>`. No `--force`, no `reset --hard`,
  no borrar branches sin pedir.

### Cosas a NO hacer / preferencias del mantenedor
- **No** proponer servicios de pago con billing internacional (Yandex Cloud,
  DeepL Pro con tarjeta, etc.): las tarjetas AR suelen ser rechazadas.
  Default para traducciones de **títulos** = `deep-translator` (Google free).
  Default para traducciones de **subtítulos** = subagente Sonnet 4.6.
- **No** re-segmentar SRT al traducir.
- **No** "limpiar" el ruso original aunque suene mal.
- **No** usar git de forma destructiva sin pedir.

### Contexto rápido del dominio
Ruslan es un youtuber ruso que arma generadores, bobinas Kacher (Качер
Бровина), réplicas de Daly, "free energy" devices y configuraciones tipo
Kulabukhov. Habla informal, repite muletillas, salta entre temas. El público
de la traducción son makers/curiosos de habla inglesa/alemana/española/italiana
que quieren replicar lo que muestra.
