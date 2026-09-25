#!/usr/bin/env python3
"""Test dello scanner del catalogo (tools/scan_models.py).

Nessun modello reale: si generano header GGUF validi e si usa `truncate` per
simulare il peso su disco, che cosi' resta sparso (0 byte reali) e i test
girano in millisecondi anche con modelli da 30 GiB.

Esecuzione:
    python3 -m unittest discover -s tests -v
"""

import json
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import scan_models as sm

MIB = 1024 * 1024


def write_gguf(path, kv, weights_bytes=0, version=3, tensor_count=0, raw_magic=b"GGUF"):
    """Scrive un GGUF sintetico: header KV vero, peso simulato con truncate."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(raw_magic)
        fh.write(struct.pack("<I", version))
        fh.write(struct.pack("<Q", tensor_count))
        fh.write(struct.pack("<Q", len(kv)))
        for key, value in kv.items():
            kb = key.encode("utf-8")
            fh.write(struct.pack("<Q", len(kb)))
            fh.write(kb)
            if isinstance(value, bool):
                vtype, payload = 7, struct.pack("<?", value)
            elif isinstance(value, str):
                vb = value.encode("utf-8")
                vtype, payload = 8, struct.pack("<Q", len(vb)) + vb
            elif isinstance(value, int):
                vtype, payload = 10, struct.pack("<Q", value)
            elif isinstance(value, float):
                vtype, payload = 6, struct.pack("<f", value)
            else:
                raise TypeError("tipo non gestito: %r" % type(value))
            fh.write(struct.pack("<I", vtype))
            fh.write(payload)
        if weights_bytes:
            fh.truncate(weights_bytes)
    return path


def base_kv(arch="llama", n_layer=32, n_kv_heads=8, head_dim=128, ctx_train=131072):
    return {
        "general.architecture": arch,
        "general.name": "Test Model",
        "llama.block_count": n_layer,
        "llama.attention.head_count": 32,
        "llama.attention.head_count_kv": n_kv_heads,
        "llama.attention.key_length": head_dim,
        "llama.attention.value_length": head_dim,
        "llama.embedding_length": 32 * head_dim,
        "llama.context_length": ctx_train,
    }


class TempTree(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.config = os.path.join(self.root, "models.json")

    def tearDown(self):
        self._tmp.cleanup()

    def make_model(self, rel, weights_mib, kv=None, **kw):
        path = os.path.join(self.root, "models", rel)
        return write_gguf(path, kv or base_kv(**kw), weights_bytes=weights_mib * MIB)


class TestParsing(TempTree):
    def test_header_letto_e_fatti_calcolati(self):
        path = self.make_model("Publisher/Model-A/Model-A-Q4_K_M.gguf", 2048)
        meta = sm.parse_gguf_header(path)
        self.assertEqual(meta["general.architecture"], "llama")
        self.assertEqual(meta["llama.block_count"], 32)
        facts = sm.model_facts(meta, os.path.getsize(path), ctx=131072)
        self.assertEqual(facts["n_ctx_train"], 131072)
        # 32 layer * 8 kv heads * (128+128) * 0.5625 = 36864 byte/token
        self.assertEqual(facts["kv_bytes_per_token"], 36864)

    def test_kv_cache_rispetta_quantizzazione(self):
        meta = base_kv()
        with open(os.path.join(self.root, "x.gguf"), "wb") as fh:
            pass
        facts_q4 = sm.model_facts(meta, 1024, ctx=1000, cache_type="q4_0")
        facts_f16 = sm.model_facts(meta, 1024, ctx=1000, cache_type="f16")
        self.assertGreater(facts_f16["kv_bytes_per_token"], facts_q4["kv_bytes_per_token"])
        self.assertEqual(facts_f16["kv_bytes_per_token"],
                         int(32 * 8 * 256 * 2.0))

    def test_header_non_gguf_solleva(self):
        path = os.path.join(self.root, "models", "broken.gguf")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(b"NOPE" + b"\x00" * 64)
        with self.assertRaises(sm.GgufError):
            sm.parse_gguf_header(path)

    def test_file_troncato_solleva(self):
        path = os.path.join(self.root, "models", "short.gguf")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(b"GGUF")
        with self.assertRaises(sm.GgufError):
            sm.parse_gguf_header(path)

    def test_versione_non_supportata(self):
        path = write_gguf(os.path.join(self.root, "v9.gguf"), base_kv(), version=9)
        with self.assertRaises(sm.GgufError):
            sm.parse_gguf_header(path)


class TestIdentity(TempTree):
    def test_id_alias_label_quant(self):
        ident = sm.derive_identity(
            "D:/Models/lmstudio/JetBrains/Mellum2-12B-A2.5B-Thinking-GGUF-Q4_K_M/"
            "Mellum2-12B-A2.5B-Thinking-Q4_K_M.gguf")
        self.assertEqual(ident["quant"], "Q4_K_M")
        self.assertEqual(ident["id"], "mellum2-12b-a2.5b-thinking")
        self.assertNotIn("GGUF", ident["label"].upper())
        self.assertIn("Mellum2", ident["alias"])

    def test_quant_riconosciute(self):
        for name, expected in [
            ("M-8B-IQ2_XS.gguf", "IQ2_XS"),
            ("M-8B-Q8_0.gguf", "Q8_0"),
            ("M-8B-BF16.gguf", "BF16"),
            ("M-8B-F16.gguf", "F16"),
        ]:
            self.assertEqual(sm.derive_identity(name)["quant"], expected, name)


class TestPlacement(TempTree):
    def test_singola_gpu_sotto_soglia(self):
        # 2 GiB pesi + 4.6 GiB KV + 1 GiB slack = 7.6 GiB -> una scheda
        path = self.make_model("P/Modello-Piccolo/Modello-Piccolo-Q4_K_M.gguf", 2048)
        mid, entry = sm.inspect_file(path)
        self.assertEqual(entry["placement_reason"], "single_gpu")
        self.assertEqual(entry["device"], ["Vulkan0"])
        self.assertLess(entry["need_mib"], 11500)

    def test_due_gpu_fra_le_soglie(self):
        path = self.make_model("P/Modello-Mezzo/Modello-Mezzo-Q4_K_M.gguf", 15360)
        mid, entry = sm.inspect_file(path)
        self.assertEqual(entry["placement_reason"], "dual_gpu")
        self.assertEqual(entry["device"], ["Vulkan0", "Vulkan1"])
        self.assertGreater(entry["need_mib"], 11500)
        self.assertLessEqual(entry["need_mib"], 22000)

    def test_troppo_grande_marcato(self):
        path = self.make_model("P/Modello-Enorme/Modello-Enorme-Q4_K_M.gguf", 30720)
        mid, entry = sm.inspect_file(path)
        self.assertEqual(entry["placement_reason"], "too_big")
        self.assertEqual(entry["device"], [])
        self.assertIn("troppo grande", entry["missing_reason"])

    def test_soglie_configurabili(self):
        # alzando la soglia single, il modello "mezzo" diventa single
        path = self.make_model("P/Modello-Mezzo/Modello-Mezzo-Q4_K_M.gguf", 15360)
        mid, entry = sm.inspect_file(path, placement={"single_max_mib": 40000})
        self.assertEqual(entry["placement_reason"], "single_gpu")

    def test_contesto_nativo_piu_basso_non_cambia_il_placement(self):
        # un modello con ctx_train piccolo usa comunque il ctx richiesto dal catalogo
        path = self.make_model("P/Seed/Seed-Q4_K_M.gguf", 512, ctx_train=4096)
        mid, entry = sm.inspect_file(path)
        self.assertEqual(entry["n_ctx_train"], 4096)
        self.assertEqual(entry["ctx"], 131072)


class TestDiscovery(TempTree):
    def test_scansione_trova_i_modelli(self):
        self.make_model("A/Alpha/Alpha-Q4_K_M.gguf", 1024)
        self.make_model("B/Beta/Beta-Q8_0.gguf", 2048)
        found = sm.scan([os.path.join(self.root, "models")])
        self.assertEqual(len(found), 2)
        self.assertIn("alpha", found)
        self.assertIn("beta", found)
        self.assertTrue(all(e["present"] for e in found.values()))

    def test_download_parziali_ignorati(self):
        self.make_model("A/Alpha/Alpha-Q4_K_M.gguf", 1024)
        partial = os.path.join(self.root, "models", "A", "Alpha", "Nuovo-Q4_K_M.gguf.part")
        with open(partial, "wb") as fh:
            fh.write(b"GGUF")
        found = sm.scan([os.path.join(self.root, "models")])
        self.assertEqual(list(found), ["alpha"])

    def test_file_corrotto_segnalato_non_esplode(self):
        self.make_model("A/Alpha/Alpha-Q4_K_M.gguf", 1024)
        bad = os.path.join(self.root, "models", "A", "Rotto", "Rotto-Q4_K_M.gguf")
        os.makedirs(os.path.dirname(bad), exist_ok=True)
        with open(bad, "wb") as fh:
            fh.write(b"XXXX" + b"\x00" * 32)
        found = sm.scan([os.path.join(self.root, "models")])
        self.assertFalse(found["rotto"]["present"])
        self.assertIn("illeggibile", found["rotto"]["missing_reason"])

    def test_cartelle_ignorate(self):
        self.make_model("A/Alpha/Alpha-Q4_K_M.gguf", 1024)
        self.make_model("node_modules/pkg/Dep-Q4_K_M.gguf", 1024)
        found = sm.scan([os.path.join(self.root, "models")])
        self.assertEqual(list(found), ["alpha"])

    def test_root_inesistente_non_esplode(self):
        self.assertEqual(sm.scan([os.path.join(self.root, "non-esiste")]), {})


class TestConfigAndDelta(TempTree):
    def setUp(self):
        super().setUp()
        self.models_root = os.path.join(self.root, "models")
        self.make_model("A/Alpha/Alpha-Q4_K_M.gguf", 1024)
        self.make_model("B/Beta/Beta-Q8_0.gguf", 15360)
        cfg = sm.load_config(self.config)[0]
        cfg["roots"] = [self.models_root]
        sm.save_config_atomic(self.config, cfg)

    def test_scrittura_atomica_lascia_json_valido(self):
        sm.rescan(self.config)
        with open(self.config, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(data["schema"], sm.SCHEMA_VERSION)
        self.assertEqual(len(data["models"]), 2)
        leftovers = [n for n in os.listdir(self.root) if n.startswith(".models-")]
        self.assertEqual(leftovers, [])

    def test_delta_aggiunta_rimozione_modifica(self):
        sm.rescan(self.config)
        # aggiunta
        self.make_model("C/Gamma/Gamma-Q4_K_M.gguf", 2048)
        result, _ = sm.rescan(self.config)
        self.assertEqual(result["delta"]["added"], ["gamma"])
        self.assertTrue(result["changed"])
        # modifica (cambia la dimensione: re-download)
        self.make_model("C/Gamma/Gamma-Q4_K_M.gguf", 3000)
        result, _ = sm.rescan(self.config)
        self.assertEqual(result["delta"]["changed"], ["gamma"])
        # rimozione
        os.remove(os.path.join(self.models_root, "A", "Alpha", "Alpha-Q4_K_M.gguf"))
        result, cfg = sm.rescan(self.config)
        self.assertEqual(result["delta"]["removed"], ["alpha"])
        self.assertFalse(cfg["models"]["alpha"]["present"])
        self.assertIn("assente", cfg["models"]["alpha"]["missing_reason"])

    def test_nessun_cambiamento_non_riscrive(self):
        sm.rescan(self.config)
        first = os.path.getmtime(self.config)
        result, _ = sm.rescan(self.config)
        self.assertFalse(result["changed"])
        self.assertEqual(os.path.getmtime(self.config), first)

    def test_voci_locked_preservate(self):
        sm.rescan(self.config)
        cfg = sm.load_config(self.config)[0]
        cfg["models"]["beta"]["locked"] = True
        cfg["models"]["beta"]["ctx"] = 32768
        cfg["models"]["beta"]["cuda"] = ["Vulkan1"]
        cfg["models"]["beta"]["sampling"] = {"temp": 0.2}
        cfg["models"]["beta"]["residency"] = "always"
        sm.save_config_atomic(self.config, cfg)

        self.make_model("B/Beta/Beta-Q8_0.gguf", 16000)  # dimensione cambiata
        _, cfg = sm.rescan(self.config)
        beta = cfg["models"]["beta"]
        self.assertEqual(beta["ctx"], 32768)
        self.assertEqual(beta["cuda"], ["Vulkan1"])
        self.assertEqual(beta["sampling"], {"temp": 0.2})
        self.assertEqual(beta["residency"], "always")
        self.assertEqual(beta["size_mib"], 16000)
        self.assertTrue(beta["present"])

    def test_campi_ignoti_dal_manager_conservati(self):
        sm.rescan(self.config)
        cfg = sm.load_config(self.config)[0]
        cfg["models"]["alpha"]["residency"] = "on_demand"
        cfg["models"]["alpha"]["stats"] = {"calls": 7}
        sm.save_config_atomic(self.config, cfg)
        self.make_model("C/Gamma/Gamma-Q4_K_M.gguf", 2048)  # forza una riscrittura
        _, cfg = sm.rescan(self.config)
        self.assertEqual(cfg["models"]["alpha"]["residency"], "on_demand")
        self.assertEqual(cfg["models"]["alpha"]["stats"], {"calls": 7})

    def test_misura_batte_la_stima(self):
        self.make_model("Publisher/MoE-12B-A2.5B-Thinking-Q4_K_M.gguf", 7153)
        sm.rescan(self.config)
        cfg, _ = sm.load_config(self.config)
        mid = "moe-12b-a2.5b-thinking"
        self.assertEqual(cfg["models"][mid]["placement_reason"], "dual_gpu")
        cfg["models"][mid]["measured"] = {"device": ["Vulkan0"], "peak_mib": 8328}
        sm.save_config_atomic(self.config, cfg)
        sm.rescan(self.config)
        cfg, _ = sm.load_config(self.config)
        self.assertEqual(cfg["models"][mid]["device"], ["Vulkan0"])
        self.assertEqual(cfg["models"][mid]["placement_reason"], "measured")

    def test_catalogo_corrotto_non_esplode(self):
        with open(self.config, "w", encoding="utf-8") as fh:
            fh.write("{non-json")
        cfg, err = sm.load_config(self.config)
        self.assertIsNotNone(err)  # default usabile, non un crash
        self.assertEqual(cfg["models"], {})
        result, _ = sm.rescan(self.config)
        self.assertIn("error", result)  # senza root non indovina
        result, cfg = sm.rescan(self.config, roots=[self.models_root])
        self.assertEqual(result["scanned"], 2)
        self.assertTrue(cfg["models"]["alpha"]["present"])

    def test_dry_run_non_scrive(self):
        sm.rescan(self.config)
        before = open(self.config, "rb").read()
        self.make_model("C/Gamma/Gamma-Q4_K_M.gguf", 2048)
        result, _ = sm.rescan(self.config, dry_run=True)
        self.assertTrue(result["changed"])
        self.assertEqual(open(self.config, "rb").read(), before)

    def test_nessuna_root_errore_esplicito(self):
        empty = os.path.join(self.root, "vuoto.json")
        result, _ = sm.rescan(empty)
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)