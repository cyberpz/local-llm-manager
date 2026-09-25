#!/usr/bin/env python3
"""Scansiona le cartelle dei modelli e mantiene il catalogo esterno (models.json).

Perche' esiste
--------------
Il catalogo era hardcoded dentro il manager: ogni modello nuovo, spostato o
cancellato richiedeva un edit del codice e un riavvio del servizio. Qui il
catalogo vive in un file esterno, prodotto da una scansione delle cartelle, e il
manager lo rilegge a caldo: niente riavvii.

Cosa ricava da solo
-------------------
- id, alias, label, quantizzazione: da nome di file e cartella
- peso su disco: dalla dimensione del file
- KV cache per token e contesto nativo: dall'header GGUF (block_count,
  attention.head_count_kv, key/value_length, context_length)
- placement: 1 GPU se ci sta in una scheda, 2 GPU se serve lo split, altrimenti
  `unsupported` con motivazione

Cosa non tocca mai
------------------
Le voci `locked: true` (tarature a mano di ctx/cuda/sampling): aggiorna solo
dimensione e presenza, cosi' una ri-scansione non cancella il lavoro manuale.

Uso
---
    python tools/scan_models.py --config models.json            # scan + delta
    python tools/scan_models.py --config models.json --dry-run   # mostra e basta
    python tools/scan_models.py --config models.json --root D:/Altro
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import tempfile
import time
from datetime import datetime, timezone

GGUF_MAGIC = b"GGUF"
SCHEMA_VERSION = 1

# Tipi di valore GGUF -> (struct format, nome descrittivo)
_GGUF_SCALARS = {
    0: "B",   # uint8
    1: "b",   # int8
    2: "H",   # uint16
    3: "h",   # int16
    4: "I",   # uint32
    5: "i",   # int32
    6: "f",   # float32
    7: "?",   # bool
    9: None,  # array (gestito a parte)
    10: "Q",  # uint64
    11: "q",  # int64
    12: "d",  # float64
}
_GGUF_STRING = 8

# Soglie di placement: configurabili dal JSON, questi sono i default misurati.
DEFAULT_PLACEMENT = {
    "single_max_mib": 11500,   # scheda pulita: 12 GB, 8328 MiB usati dal 12B a 131k
    "dual_max_mib": 22000,     # due schede: il 35B a 131k sta in 22,15 GiB
    "slack_mib": 1024,         # buffer di compute e frammentazione
    "single_devices": ["Vulkan0"],            # scheda fisica 1 (pulita) in llama.cpp
    "dual_devices": ["Vulkan0", "Vulkan1"],
    "default_ctx": 131072,
    "cache_type": "q4_0",
}

# Byte per elemento della KV cache per tipo di quantizzazione.
_CACHE_BYTES_PER_ELEM = {
    "f16": 2.0,
    "q8_0": 1.0625,
    "q5_1": 0.75,
    "q5_0": 0.6875,
    "q4_1": 0.625,
    "q4_0": 0.5625,
    "iq4_nl": 0.5625,
}

_QUANT_RE = re.compile(
    r"(IQ\d+_[A-Z0-9]+|Q\d+_K_[A-Z]+|Q\d+_K|Q\d+_\d+|BF16|F16|F32)", re.IGNORECASE
)

DEFAULT_IGNORE_DIRS = (
    "$recycle.bin", "system volume information", "windows", "appdata",
    "node_modules", "site-packages", "__pycache__", "logs", ".git",
    ".cache", "temp", "tmp",
)


class GgufError(Exception):
    """Header GGUF illeggibile o non-GGUF."""


class _Reader:
    """Lettore sequenziale minimale su file binario."""

    def __init__(self, fh):
        self.fh = fh

    def raw(self, n):
        data = self.fh.read(n)
        if len(data) != n:
            raise GgufError("file troncato durante la lettura dell'header")
        return data

    def unpack(self, fmt):
        size = struct.calcsize("<" + fmt)
        return struct.unpack("<" + fmt, self.raw(size))[0]

    def string(self):
        n = self.unpack("Q")
        if n > 64 * 1024 * 1024:
            raise GgufError("stringa di metadata irragionevole (%d byte)" % n)
        return self.raw(n).decode("utf-8", errors="replace")


def _read_value(reader, vtype):
    if vtype == _GGUF_STRING:
        return reader.string()
    if vtype == 9:  # array
        elem_type = reader.unpack("I")
        count = reader.unpack("Q")
        if count > 10_000_000:
            raise GgufError("array di metadata irragionevole (%d elementi)" % count)
        return [_read_value(reader, elem_type) for _ in range(count)]
    if vtype not in _GGUF_SCALARS:
        raise GgufError("tipo di metadata sconosciuto: %s" % vtype)
    return reader.unpack(_GGUF_SCALARS[vtype])


def parse_gguf_header(path, keys_wanted=None, max_kv=4096):
    """Legge l'header GGUF e restituisce i metadata come dizionario piatto.

    Si ferma appena ha raccolto tutte le chiavi richieste: negli header GGUF i
    metadata `general.*` e `<arch>.*` vengono prima dei grandi array del
    tokenizer, quindi in pratica si leggono poche centinaia di byte.
    """
    wanted = set(keys_wanted or ())
    meta = {}
    with open(path, "rb") as fh:
        reader = _Reader(fh)
        if reader.raw(4) != GGUF_MAGIC:
            raise GgufError("magic GGUF assente")
        version = reader.unpack("I")
        if version not in (2, 3):
            raise GgufError("versione GGUF non supportata: %s" % version)
        tensor_count = reader.unpack("Q")
        kv_count = reader.unpack("Q")
        if kv_count > 1_000_000:
            raise GgufError("conteggio metadata irragionevole: %s" % kv_count)
        for _ in range(min(kv_count, max_kv)):
            key = reader.string()
            vtype = reader.unpack("I")
            value = _read_value(reader, vtype)
            if isinstance(value, list):
                meta[key] = "[lista di %d]" % len(value)
            else:
                meta[key] = value
            if wanted and wanted.issubset(meta.keys()):
                break
    meta["_gguf_version"] = version
    meta["_tensor_count"] = tensor_count
    meta["_kv_count"] = kv_count
    return meta


def arch_of(meta):
    return meta.get("general.architecture") or ""


def model_facts(meta, file_bytes, ctx=None, cache_type="q4_0"):
    """Ricava contesto nativo, KV per token e peso dai metadata + dimensione file."""
    arch = arch_of(meta)

    def g(key, default=None):
        return meta.get(arch + "." + key, meta.get(key, default))

    n_layer = g("block_count")
    n_kv_heads = g("attention.head_count_kv", g("attention.head_count"))
    head_count = g("attention.head_count")
    k_len = g("attention.key_length")
    v_len = g("attention.value_length")
    if k_len is None and head_count:
        k_len = g("embedding_length", 0) // head_count if g("embedding_length") else None
    if v_len is None:
        v_len = k_len

    n_ctx_train = g("context_length")
    bpe = _CACHE_BYTES_PER_ELEM.get(cache_type, 0.5625)

    kv_per_token = None
    if n_layer and n_kv_heads and k_len and v_len:
        kv_per_token = int(n_layer * n_kv_heads * (k_len + v_len) * bpe)

    return {
        "arch": arch,
        "n_layer": n_layer,
        "n_kv_heads": n_kv_heads,
        "n_ctx_train": n_ctx_train,
        "kv_bytes_per_token": kv_per_token,
        "weights_bytes": file_bytes,
        "cache_type": cache_type,
        "ctx": ctx,
    }


def estimate_vram_mib(weights_bytes, kv_per_token, ctx, slack_mib):
    kv = (kv_per_token or 0) * ctx
    return int(round((weights_bytes + kv) / (1024 * 1024))) + slack_mib


def choose_placement(weights_bytes, kv_per_token, ctx, placement=None):
    """Decide device list e motivazione. Restituisce (devices, reason, need_mib)."""
    cfg = dict(DEFAULT_PLACEMENT)
    cfg.update(placement or {})
    need = estimate_vram_mib(weights_bytes, kv_per_token, ctx, cfg["slack_mib"])
    if need <= cfg["single_max_mib"]:
        return list(cfg["single_devices"]), "single_gpu", need
    if need <= cfg["dual_max_mib"]:
        return list(cfg["dual_devices"]), "dual_gpu", need
    return [], "too_big", need


def derive_identity(path, root=None):
    """Ricava id, alias, label e quantizzazione da nome file e cartella."""
    stem = os.path.splitext(os.path.basename(path))[0]
    quant_match = _QUANT_RE.search(stem)
    quant = quant_match.group(0).upper() if quant_match else None
    base = _QUANT_RE.sub("", stem).strip(" -_.")
    slug = re.sub(r"[^a-z0-9.]+", "-", base.lower()).strip("-.")
    human = re.sub(r"[_-]+", " ", base).strip()
    human = re.sub(r"\s{2,}", " ", human)
    # La cartella contenitore e' spesso piu' leggibile del nome file
    parent = os.path.basename(os.path.dirname(path))
    parent_base = _QUANT_RE.sub("", parent).strip(" -_.")
    label = human or re.sub(r"[_-]+", " ", parent_base)
    if "GGUF" in label.upper():
        label = re.sub(r"\s*GGUF\s*", " ", label, flags=re.IGNORECASE).strip()
    return {
        "id": slug,
        "alias": human or parent_base,
        "label": label,
        "quant": quant,
        "parent": parent,
    }


def iter_gguf_files(roots, ignore_dirs=DEFAULT_IGNORE_DIRS):
    """Cammina le root e restituisce i .gguf completi (salta download parziali)."""
    for root in roots:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
            dirnames[:] = [
                d for d in dirnames
                if d.lower() not in ignore_dirs and not d.startswith(".")
            ]
            for name in sorted(filenames):
                low = name.lower()
                if not low.endswith(".gguf"):
                    continue
                if low.endswith((".gguf.part", ".gguf.tmp", ".gguf.download")):
                    continue
                yield os.path.join(dirpath, name)


def inspect_file(path, placement=None, ctx_override=None, keys_wanted=None, previous=None):
    """Ispeziona un singolo .gguf: identita', fatti, placement. Non solleva mai."""
    ident = derive_identity(path)
    entry = {
        "path": path,
        "size_bytes": None,
        "ctx": None,
        "device": [],
        "placement_reason": None,
        "need_mib": None,
        "arch": None,
        "n_ctx_train": None,
        "kv_bytes_per_token": None,
        "quant": ident["quant"],
        "alias": ident["alias"],
        "label": ident["label"],
        "present": True,
        "missing_reason": None,
    }
    try:
        entry["size_bytes"] = os.path.getsize(path)
    except OSError as exc:
        entry["present"] = False
        entry["missing_reason"] = "stat fallita: %s" % exc
        return ident["id"], entry

    wanted = keys_wanted or {
        "general.architecture",
        "general.name",
        "general.file_type",
    }
    try:
        meta = parse_gguf_header(path, keys_wanted=None)
    except (GgufError, OSError, struct.error) as exc:
        entry["present"] = False
        entry["missing_reason"] = "header illeggibile: %s" % exc
        return ident["id"], entry

    arch = arch_of(meta)
    if not arch:
        entry["present"] = False
        entry["missing_reason"] = "architettura assente nei metadata"
        return ident["id"], entry

    ctx = ctx_override or DEFAULT_PLACEMENT["default_ctx"]
    facts = model_facts(meta, entry["size_bytes"], ctx=ctx,
                        cache_type=(placement or DEFAULT_PLACEMENT).get(
                            "cache_type", DEFAULT_PLACEMENT["cache_type"]))
    devices, reason, need = choose_placement(
        facts["weights_bytes"], facts["kv_bytes_per_token"], ctx, placement
    )
    entry.update({
        "ctx": ctx,
        "device": devices,
        "placement_reason": reason,
        "need_mib": need,
        "arch": arch,
        "n_ctx_train": facts["n_ctx_train"],
        "kv_bytes_per_token": facts["kv_bytes_per_token"],
        "size_mib": int(round(entry["size_bytes"] / (1024 * 1024))),
    })
    prev = _previous_for(previous, ident["id"], path)
    measured = (prev or {}).get("measured") or {}
    if measured.get("device"):
        # La misura batte la stima: sliding-window e MoE ibridi sballano la formula KV
        # (Mellum2 12B: stimati 11633 MiB, misurati 8328 MiB su una scheda sola).
        entry["device"] = [str(d) for d in measured["device"]]
        entry["placement_reason"] = "measured"
        if measured.get("peak_mib"):
            entry["need_mib"] = int(measured["peak_mib"])
        if measured.get("ctx_ok"):
            entry["ctx_ok"] = int(measured["ctx_ok"])
    if not entry["device"]:
        entry["present"] = True
        entry["missing_reason"] = "troppo grande per 2 schede: %d MiB stimati" % need
    return ident["id"], entry


def _previous_for(previous, model_id, path):
    """Trova la voce precedente per id, poi per nome file (sopravvive a uno spostamento)."""
    if not previous:
        return None
    if model_id in previous:
        return previous[model_id]
    base = os.path.basename(path).lower()
    for entry in previous.values():
        if os.path.basename(str(entry.get("path") or "")).lower() == base:
            return entry
    return None


def scan(roots, placement=None, ignore_dirs=DEFAULT_IGNORE_DIRS, previous=None):
    """Scansiona le root e restituisce {id: entry}. `previous` serve per le misure note."""
    found = {}
    for path in iter_gguf_files(roots, ignore_dirs=ignore_dirs):
        model_id, entry = inspect_file(path, placement=placement, previous=previous)
        # Collisione di id: si disambigua col nome della cartella contenitrice
        if model_id in found:
            suffix = re.sub(r"[^a-z0-9]+", "-",
                            os.path.basename(os.path.dirname(path)).lower()).strip("-")
            model_id = "%s-%s" % (model_id, suffix) if suffix else model_id + "-2"
        found[model_id] = entry
    return found


def diff_catalog(old_models, new_models):
    """Delta fra catalogo esistente e scansione: added/removed/changed/unchanged."""
    old_ids, new_ids = set(old_models or {}), set(new_models or {})
    added = sorted(new_ids - old_ids)
    removed = sorted(old_ids - new_ids)
    changed, unchanged = [], []
    for mid in sorted(old_ids & new_ids):
        old, new = old_models[mid], new_models[mid]
        if (old.get("size_bytes") != new.get("size_bytes")
                or os.path.normcase(old.get("path", "")) != os.path.normcase(new.get("path", ""))
                or list(old.get("device") or []) != list(new.get("device") or [])):
            changed.append(mid)
        else:
            unchanged.append(mid)
    return {"added": added, "removed": removed, "changed": changed,
            "unchanged": unchanged}


def merge_catalog(existing_models, scanned, placement=None):
    """Unisce la scansione al catalogo esistente rispettando le voci `locked`."""
    merged = {}
    for mid, entry in scanned.items():
        old = (existing_models or {}).get(mid)
        if old and old.get("locked"):
            keep = dict(old)
            keep["size_bytes"] = entry["size_bytes"]
            keep["size_mib"] = entry.get("size_mib")
            keep["present"] = entry["present"]
            keep["missing_reason"] = entry["missing_reason"]
            keep["path"] = old.get("path") or entry["path"]
            merged[mid] = keep
            continue
        new = dict(entry)
        if old:
            # conserva i campi che non sono competenza dello scanner
            for field in ("residency", "sampling", "notes", "last_used", "stats"):
                if field in old:
                    new[field] = old[field]
        if placement and "cache_type" in placement:
            new.setdefault("sampling", {})
        merged[mid] = new

    # le voci sparite dal disco restano nel catalogo, marcate
    for mid, old in (existing_models or {}).items():
        if mid in merged:
            continue
        gone = dict(old)
        gone["present"] = False
        gone["missing_reason"] = gone.get("missing_reason") or "assente dal disco"
        gone["size_bytes"] = None
        gone["size_mib"] = None
        merged[mid] = gone
    return merged


def load_config(path):
    """Legge il catalogo esterno; se manca o e' rotto restituisce la struttura minima."""
    base = {
        "schema": SCHEMA_VERSION,
        "generated_at": None,
        "roots": [],
        "placement": dict(DEFAULT_PLACEMENT),
        "models": {},
    }
    if not path or not os.path.exists(path):
        return base, None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        return base, "catalogo illeggibile: %s" % exc
    if not isinstance(data, dict):
        return base, "catalogo non e' un oggetto JSON"
    base.update({k: v for k, v in data.items() if k != "models"})
    models = data.get("models")
    base["models"] = models if isinstance(models, dict) else {}
    return base, None


def save_config_atomic(path, config):
    """Scrive il catalogo con tmp + replace: mai un file a meta'."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".models-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, ensure_ascii=False, sort_keys=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def rescan(config_path, roots=None, dry_run=False, verbose=False):
    """Ciclo completo: legge, scansiona, unisce, scrive se e' cambiato qualcosa."""
    cfg, err = load_config(config_path)
    if err:
        print("ATTENZIONE: %s" % err, file=sys.stderr)
    roots = roots or cfg.get("roots") or []
    if not roots:
        return {"error": "nessuna root da scansionare", "delta": None}, cfg
    placement = cfg.get("placement") or DEFAULT_PLACEMENT
    existing = cfg.get("models") or {}
    scanned = scan(roots, placement=placement, previous=existing)
    delta = diff_catalog(existing, scanned)
    merged = merge_catalog(existing, scanned, placement=placement)

    if verbose:
        for mid in delta["added"]:
            print("  + %s" % mid)
        for mid in delta["removed"]:
            print("  - %s" % mid)
        for mid in delta["changed"]:
            print("  ~ %s" % mid)

    changed = bool(delta["added"] or delta["removed"] or delta["changed"])
    if changed and not dry_run:
        cfg["models"] = merged
        cfg["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_config_atomic(config_path, cfg)
    return {"delta": delta, "scanned": len(scanned), "changed": changed}, cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description="Scansiona i modelli e aggiorna models.json")
    ap.add_argument("--config", default="models.json", help="percorso del catalogo esterno")
    ap.add_argument("--root", action="append", default=None,
                    help="root da scansionare (ripetibile; default: quelle nel JSON)")
    ap.add_argument("--dry-run", action="store_true", help="mostra il delta senza scrivere")
    ap.add_argument("--list", action="store_true", help="stampa il catalogo risultante")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    started = time.time()
    result, cfg = rescan(args.config, roots=args.root, dry_run=args.dry_run,
                         verbose=True)  # il CLI stampa sempre il delta
    if result.get("error"):
        print("ERRORE: %s" % result["error"], file=sys.stderr)
        return 2
    delta = result["delta"]
    print("scansionati: %d modelli in %.2fs" % (result["scanned"], time.time() - started))
    print("delta: +%d -%d ~%d (invariati %d)" % (
        len(delta["added"]), len(delta["removed"]), len(delta["changed"]),
        len(delta["unchanged"])))
    if result["changed"]:
        print("catalogo aggiornato" + (" (dry-run: non scritto)" if args.dry_run else ""))
    else:
        print("nessun cambiamento")
    if args.list:
        models = cfg.get("models") or {}
        for mid in sorted(models):
            m = models[mid]
            print("  %-32s %6s MiB  %-28s %s" % (
                mid, m.get("size_mib") or "-", ",".join(m.get("device") or []) or "-",
                m.get("placement_reason") or "-"))
    return 0


if __name__ == "__main__":
    sys.exit(main())