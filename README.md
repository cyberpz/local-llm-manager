# local-llm-manager

Selettore modelli locale per **MasterBeef** (2× RTX 3060 12 GB, Vulkan).
Gira come attività pianificata `GiorgioModelManager` (utente SYSTEM).

## Layout in produzione — `C:\Users\Peppuz\masterbeef-llm\`

```
local_llm_manager.py     script servito dall'attività pianificata
domus_toggle.bat         toggle di servizi + warmup LLM
tools/                   utility (migrazione path modelli, benchmark)
bench/                   script e risultati dei bench di placement
logs/                    manager.log, manager-error.log, llama-server-*.log   (non versionato)
state/                   local-llm-manager-state.json                        (non versionato)
bak/                     backup delle versioni precedenti                     (non versionato)
```

## Porte

| Porta | Ruolo |
|-------|-------|
| 1234 | PUBLIC — OpenAI-compatibile. `/v1/models` espone **sempre tutto il catalogo**, anche a scatola fredda; i modelli si caricano on demand |
| 1235 | ADMIN / back-compat — stesso handler, target del tunnel autossh verso la VPS (AIProxy) |
| 1236 | interno — llama.cpp (`llama.exe serve`) |

`/v1/chat/completions` richiede `Authorization: Bearer giorgio-local-manager`.
`/v1/models`, `/v1/models/{id}`, `/health`, `/status` sono aperti (discovery prima del load).

Per ogni modello: `status` = `idle | loading | ready | error | missing | unsupported`,
`loaded`, `load_on_demand`, `unsupported_reason`.

## Riavvio

```powershell
$p = (Get-CimInstance Win32_Process -Filter "name='python.exe'" |
      Where-Object CommandLine -like '*local_llm_manager*').ProcessId
taskkill /F /PID $p
schtasks /run /tn GiorgioModelManager
```

All'avvio: libera la porta pubblica da eventuali llama orfani (layout v3), poi **adotta** un
llama già vivo su 1236 (conservando `effective_ctx` dal file di stato) o ripristina l'ultimo
modello dal file di stato.

## Regole di placement (misurate, llama-bench build f04801018/10078)

- Modello che entra in **una** scheda → **una** scheda (`-sm none`): +7% (12B) e +12% (2B) in tg
  rispetto allo split su due GPU. Scheda preferita: **Vulkan1** (più libera; la 0 porta il desktop).
- `-sm row` **non carica** su Vulkan con questi GGUF (`failed to load model`).
- Split a 2 GPU solo per i 35B A3B (`-sm layer`, ~20.2 GiB di pesi).
- Modelli con architettura non supportata vanno marcati `unsupported` nel catalogo: così il
  selettore risponde 400 subito invece di bruciare l'intera ctx ladder.

## Budget VRAM osservato

| GPU | usata a riposo | libera |
|-----|----------------|--------|
| Vulkan0 | ~960 MiB (dwm 766 + explorer 206 + csrss 103) | ~11.2 GB |
| Vulkan1 | ~1 MiB | ~12.1 GB |

## Spostare i modelli su un altro disco

```bash
python tools/migrate_model_paths.py --new-root E:\Models            # dry-run
python tools/migrate_model_paths.py --new-root E:\Models --apply
```
Riscrive i `path` del catalogo e aggiunge la nuova radice a `MODEL_ROOTS`.