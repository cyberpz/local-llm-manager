#!/usr/bin/env python3
"""Riscrive i path del catalogo di local-llm-manager.py quando i modelli cambiano disco.

Uso (su MasterBeef, col manager fermo):
    python migrate_model_paths.py --new-root E:\\Models            # dry-run: mostra la mappa
    python migrate_model_paths.py --new-root E:\\Models --apply    # riscrive il file

Regole:
- per ogni voce del catalogo cerca il nome file sotto il nuovo root (ricorsivo, max 4 livelli);
- se lo trova aggiorna "path"; se non lo trova lo segnala come MISSING e non tocca la voce;
- aggiunge il nuovo root a MODEL_ROOTS se assente (il fallback root+nome resta come rete di sicurezza).
"""
import argparse
import os
import re
import shutil
import sys
import time

MANAGER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "local_llm_manager.py")


def catalog_entries(text):
    """(model_id, filename, path_attuale) per ogni voce del catalogo."""
    out = []
    for block in re.finditer(r'^\s{4}"([A-Za-z0-9._-]+)": \{(.*?)^\s{4}\},', text, re.S | re.M):
        mid, body = block.group(1), block.group(2)
        f = re.search(r'"file":\s*"([^"]+)"', body)
        p = re.search(r'"path":\s*"([^"]+)"', body)
        if f:
            # nel sorgente i path sono letterali con doppi backslash: normalizza per il confronto
            old = p.group(1).replace("\\\\", "\\") if p else None
            out.append((mid, f.group(1), old))
    return out


def as_literal(path):
    """Nel file il path va scritto come letterale Python: doppi backslash."""
    return path.replace("\\", "\\\\")


def find_under(root, filename, max_depth=4):
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if depth >= max_depth:
            dirnames[:] = []
            continue
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-root", required=True, help=r"nuovo root, es. E:\Models")
    ap.add_argument("--manager", default=MANAGER)
    ap.add_argument("--apply", action="store_true", help="scrive davvero (senza: dry-run)")
    args = ap.parse_args()

    with open(args.manager, encoding="utf-8") as fh:
        text = fh.read()
    entries = catalog_entries(text)
    if not entries:
        sys.exit("nessuna voce di catalogo trovata: il formato del file è cambiato")

    if not os.path.isdir(args.new_root):
        sys.exit(f"root non accessibile: {args.new_root}")

    hits, misses, changes = [], [], []
    for mid, filename, old_path in entries:
        found = find_under(args.new_root, filename)
        if found:
            hits.append((mid, found))
            if old_path != found:
                changes.append((mid, old_path, found))
        else:
            misses.append((mid, filename, old_path))

    print(f"trovati {len(hits)}/{len(entries)} sotto {args.new_root}")
    for mid, p in hits:
        print(f"  OK      {mid:32} {p}")
    for mid, fn, old in misses:
        print(f"  MISSING {mid:32} {fn}  (resta su {old})")

    if not changes:
        print("\nnessun path da cambiare")
        return

    print(f"\n{len(changes)} path da riscrivere:")
    for mid, old, new in changes:
        print(f"  {mid}\n    da {old}\n    a  {new}")

    if not args.apply:
        print("\ndry-run: nessuna modifica. Rilancia con --apply")
        return

    backup = f"{args.manager}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(args.manager, backup)
    for mid, old, new in changes:
        text = text.replace(f'"path": "{as_literal(old)}"', f'"path": "{as_literal(new)}"')
    if args.new_root not in text:
        text = text.replace('MODEL_ROOTS = [', f'MODEL_ROOTS = [\n    r"{args.new_root}",', 1)
    with open(args.manager, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"scritto {args.manager} (backup: {backup})")


if __name__ == "__main__":
    main()