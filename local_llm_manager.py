"""
Local LLM Manager — switch models on llama.cpp (llama.exe serve, Vulkan) with GPU memory checks.
Renames: GiorgioModelManager → Local-LLM-Manager
"""
import http.server, json, subprocess, time, threading, os, urllib.request, socket, sys
from datetime import datetime

# Windows console may be cp1252: force UTF-8 so log lines never crash the process
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Model roots for search fallback
MODEL_ROOTS = [
    "C:\\Users\\Peppuz",
    "D:\\Models",
]

MODEL_CATALOG = {
    "nemotron-cascade-2-30b-a3b": {
        "file": "nvidia_Nemotron-Cascade-2-30B-A3B-Q4_0.gguf",
        "path": "C:\\Users\\Peppuz\\.lmstudio\\models\\bartowski\\nvidia_Nemotron-Cascade-2-30B-A3B-GGUF\\nvidia_Nemotron-Cascade-2-30B-A3B-Q4_0.gguf",
        "alias": "Nemotron Cascade 2",
        "label": "Nemotron Cascade 2 — Qualità",
        "cuda": ["Vulkan0", "Vulkan1"],
        "ctx": 32768,
    },
    "qwen3.5-35b-a3b": {
        "file": "Qwen3.5-35B-A3B-Q4_K_M.gguf",
        "path": "C:\\Users\\Peppuz\\.lmstudio\\models\\unsloth\\Qwen3.5-35B-A3B-GGUF\\Qwen3.5-35B-A3B-Q4_K_M.gguf",
        "alias": "Qwen 3.5",
        "label": "Qwen 3.5 — Ragionamento",
        "cuda": ["Vulkan0", "Vulkan1"],
        "ctx": 65536,
    },
    "mellum2-12b-a2.5b": {
        "file": "Mellum2-12B-A2.5B-Thinking-Q4_K_M.gguf",
        "path": "C:\\Users\\Peppuz\\.lmstudio\\models\\JetBrains\\Mellum2-12B-A2.5B-Thinking-GGUF-Q4_K_M\\Mellum2-12B-A2.5B-Thinking-Q4_K_M.gguf",
        "alias": "Mellum 2",
        "label": "Mellum 2 — Codice e velocità",
        "cuda": ["Vulkan0"],
        "ctx": 131072,
    },
    "gemma-4-e4b": {
        "file": "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
        "path": "C:\\Users\\Peppuz\\.lmstudio\\models\\HauhauCS\\Gemma-4-E4B-Uncensored-HauhauCS-Aggressive\\Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
        "alias": "Gemma 4 E4B",
        "label": "Gemma 4 E4B — Leggero",
        "cuda": ["Vulkan0"],
        "ctx": None,
    },
    "qwen3.6-35b-a3b-uncensored": {
        "file": "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
        "path": "D:\\Models\\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
        "alias": "Qwen 3.6 Uncensored",
        "label": "Qwen 3.6 Uncensored — Benchmark",
        "cuda": ["Vulkan0", "Vulkan1"],
        "ctx": 32768,
    },
    "ornith-1.5-35b-a3b": {
        "file": "Ornith-1.5-35B-Q4_K_M.gguf",
        "path": "D:\\Models\\Ornith-1.5-35B-Q4_K_M.gguf",
        "alias": "Ornith 1.5",
        "label": "Ornith 1.5 — Benchmark",
        "cuda": ["Vulkan0", "Vulkan1"],
        "ctx": 32768,
    },
    "qwen3.8-27b-gsq-rco": {
        "file": "Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf",
        "path": "D:\\Models\\Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf",
        "alias": "Qwen 3.8 GSQ",
        "label": "Qwen 3.8 GSQ — Compatto",
        "cuda": ["Vulkan0"],
        "ctx": 32768,
    },
}

LLAMA_SERVER = "C:\\Users\\Peppuz\\AppData\\Local\\Microsoft\\WindowsApps\\llama.exe"
BASE_DIR = "C:\\Users\\Peppuz"
STATE_FILE = "C:\\Users\\Peppuz\\local-llm-manager-state.json"
LOG_FILE = "C:\\Users\\Peppuz\\local-llm-manager.log"
ERROR_LOG_FILE = "C:\\Users\\Peppuz\\local-llm-manager-error.log"
API_KEY = "giorgio-local-manager"
PORT = 1235

# Context size ladder for retry (highest to lowest)
CTX_LADDER = [262144, 131072, 65536, 32768, 16384, 8192]

state = {
    "state": "idle",
    "model_id": None,
    "alias": None,
    "error": None,
    "effective_ctx": None,
    "requested_ctx": None,
    "updated_at": None,
    "last_switch": None,
    "pid": None,
}
state_lock = threading.Lock()
switch_lock = threading.Lock()


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def log_error(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    try:
        with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def save_state():
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        log_error(f"Failed to save state: {e}")


def find_model_file(model_id):
    """Return model file path. Uses explicit 'path' if set, else searches MODEL_ROOTS."""
    spec = MODEL_CATALOG.get(model_id)
    if not spec:
        return None
    # Prefer explicit path
    if "path" in spec and os.path.isfile(spec["path"]):
        return spec["path"]
    # Fallback: search in MODEL_ROOTS
    filename = spec["file"]
    for root in MODEL_ROOTS:
        path = os.path.join(root, filename)
        if os.path.isfile(path):
            return path
    return None


def get_gpu_memory():
    """Get GPU memory using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return None
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({
                    "index": int(parts[0]),
                    "used_mib": int(parts[1]),
                    "free_mib": int(parts[2]),
                })
        return gpus
    except Exception as e:
        log_error(f"GPU query failed: {e}")
        return None


def find_llama_pids():
    """Find ALL running llama.exe / llama-server.exe PIDs (returns list)."""
    pids = []
    for image in ("llama.exe", "llama-server.exe"):
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"IMAGENAME eq {image}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.strip().split("\n"):
                if image.lower() in line.lower():
                    parts = line.split(",")
                    if len(parts) >= 2:
                        try:
                            pids.append(int(parts[1].strip('"')))
                        except ValueError:
                            pass
        except Exception:
            pass
    return sorted(set(pids))


def port_1234_busy():
    """True if something is listening on 127.0.0.1:1234."""
    try:
        with socket.create_connection(("127.0.0.1", 1234), timeout=1):
            return True
    except OSError:
        return False


def kill_llama_process():
    """Kill every running llama process, then wait for port 1234 to be released."""
    pids = find_llama_pids()
    for pid in pids:
        try:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=10)
        except Exception as e:
            log_error(f"Failed to kill process {pid}: {e}")
    if pids:
        time.sleep(2)
    # Wait until port is actually free (guards against PID-miss / slow release)
    for _ in range(10):
        if not port_1234_busy():
            return True
        time.sleep(1)
    if port_1234_busy():
        log_error("Port 1234 still busy after kill attempts")
        return False
    return True


def check_llama_health(timeout=5, expected_alias=None):
    """Check llama server health. /health returns 200 only when the model is
    loaded and ready (503 while loading), so a 200 means OUR server is up
    (do_switch kills + waits for port release before starting).
    expected_alias is a soft check: log a mismatch but never block on it,
    since the alias is cosmetic (llama serves the loaded model for any name)."""
    try:
        req = urllib.request.Request("http://127.0.0.1:1234/health", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return False
        if expected_alias:
            try:
                req2 = urllib.request.Request("http://127.0.0.1:1234/v1/models", method="GET")
                with urllib.request.urlopen(req2, timeout=timeout) as resp2:
                    data = json.loads(resp2.read().decode())
                ids = [m.get("id", "") for m in data.get("data", [])]
                ids += [m.get("id", "") for m in data.get("models", [])]
                if ids and expected_alias not in ids:
                    log(f"  note: server reports model {ids}, expected alias {expected_alias}")
            except Exception:
                pass
        return True
    except Exception:
        return False


def proc_alive(pid):
    """True if the given PID is still running (Windows tasklist)."""
    if not pid:
        return False
    try:
        r = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        # Match the quoted CSV field exactly: a bare str(pid) can false-positive
        # against the memory column (e.g. pid 352 in "63,844,352 K").
        return f'"{pid}"' in (r.stdout or "")
    except Exception:
        # If we can't tell, assume alive and let health polling decide
        return True


def wait_for_llama_ready(timeout=180, pid=None, expected_alias=None):
    """Wait until the server is healthy (and optionally serving expected_alias).

    Returns False EARLY if the child process dies (e.g. OOM at this context
    size) so we don't burn the full timeout on a guaranteed-failed attempt —
    this is what made every switch waste minutes on 262k/131k contexts."""
    start = time.time()
    while time.time() - start < timeout:
        if check_llama_health(expected_alias=expected_alias):
            return True
        if pid and not proc_alive(pid):
            log(f"  server pid {pid} exited before ready (ctx too large / crash)")
            return False
        time.sleep(1)
    return False


def start_llama_server(model_path, ctx_size, cuda_devices, alias=None):
    """Start llama.exe serve with given parameters. Returns (pid, log_path)."""
    cuda_str = ",".join(cuda_devices)
    cmd = [
        LLAMA_SERVER,
        "serve",
        "-m", model_path,
        "-c", str(ctx_size),
        "--host", "127.0.0.1",
        "--port", "1234",
        "-ngl", "999",
        "--device", cuda_str,
        "--threads", "4",
        "-fa", "on",
    ]
    if alias:
        cmd += ["--alias", alias]
    log_path = os.path.join(BASE_DIR, f"llama-server-{int(time.time())}.log")
    log(f"Starting: {' '.join(cmd)} (log: {log_path})")

    env = os.environ.copy()
    llama_dir = os.path.dirname(LLAMA_SERVER)
    env["PATH"] = llama_dir + ";" + env.get("PATH", "")

    try:
        # Capture output to a rotating-ish log file (was DEVNULL: failures were undiagnosable)
        logf = open(log_path, "ab")
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
            env=env
        )
        return proc.pid, log_path
    except Exception as e:
        log_error(f"Failed to start server: {e}")
        return None, None


def cleanup_old_logs(keep=5):
    """Keep only the N most recent llama-server-*.log files."""
    try:
        logs = [os.path.join(BASE_DIR, f) for f in os.listdir(BASE_DIR)
                if f.startswith("llama-server-") and f.endswith(".log")]
        logs.sort(key=os.path.getmtime, reverse=True)
        for old in logs[keep:]:
            try:
                os.remove(old)
            except OSError:
                pass
    except Exception:
        pass


def build_ctx_ladder(spec, requested_ctx=None):
    """Context sizes to try, highest first.

    If the caller pins requested_ctx, cap the ladder at it (respect the request).
    Otherwise try the full ladder so we keep the largest ctx that actually fits
    (e.g. qwen3.8 reached 65536 even though its catalog ctx is 32768); the
    fast-fail in wait_for_llama_ready makes the high attempts cheap."""
    if requested_ctx:
        ladder = [c for c in CTX_LADDER if c <= requested_ctx]
        if requested_ctx not in ladder:
            ladder.insert(0, requested_ctx)
        return ladder or [requested_ctx]
    return list(CTX_LADDER)


def do_switch(model_id, requested_ctx=None):
    """Perform model switch with context ladder retry."""
    spec = MODEL_CATALOG.get(model_id)
    if not spec:
        with state_lock:
            state.update({
                "state": "error", "model_id": model_id, "alias": None,
                "error": f"Unknown model: {model_id}", "pid": None,
                "updated_at": time.time(),
            })
            save_state()
        return False

    model_path = find_model_file(model_id)
    if not model_path:
        with state_lock:
            state.update({
                "state": "error", "model_id": model_id, "alias": spec["alias"],
                "error": f"Model file not found: {spec['file']} in {MODEL_ROOTS}",
                "pid": None, "updated_at": time.time(),
            })
            save_state()
        return False

    ctx_list = build_ctx_ladder(spec, requested_ctx)

    with state_lock:
        state.update({
            "state": "switching", "model_id": model_id, "alias": spec["alias"],
            "error": None, "updated_at": time.time(),
        })
        save_state()

    log(f"Switching to {spec['alias']} ({model_id}), ctx ladder: {ctx_list}")

    old_pids = find_llama_pids()
    killed_ok = kill_llama_process()
    old_process_exited = killed_ok and not (set(old_pids) & set(find_llama_pids()))

    # Sample GPU after kill (3 real samples) to document VRAM release
    release_samples = []
    for _ in range(3):
        g = get_gpu_memory()
        if g:
            release_samples.append(g)
        time.sleep(2)
    gpus_after_kill = release_samples[-1] if release_samples else None

    switch_start = time.time()
    for ctx in ctx_list:
        log(f"Trying ctx={ctx}")
        pid, log_path = start_llama_server(model_path, ctx, spec["cuda"], alias=model_id)
        if not pid:
            continue

        if wait_for_llama_ready(timeout=180, pid=pid):
            ready_seconds = round(time.time() - switch_start, 1)
            served_ids = []
            try:
                req = urllib.request.Request("http://127.0.0.1:1234/v1/models", method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    mj = json.loads(resp.read().decode())
                served_ids = [m.get("id", "") for m in (mj.get("data") or mj.get("models") or [])]
            except Exception:
                pass
            log(f"Ready: {spec['alias']} ctx={ctx} pid={pid} in {ready_seconds}s, served as {served_ids}")
            gpus_loaded = get_gpu_memory()
            with state_lock:
                state.update({
                    "state": "ready", "model_id": model_id, "alias": spec["alias"],
                    "error": None, "effective_ctx": ctx, "requested_ctx": requested_ctx,
                    "pid": pid, "updated_at": time.time(),
                    "last_switch": {
                        "released_gpu": gpus_after_kill,
                        "release_samples": release_samples,
                        "old_process_exited": old_process_exited,
                        "ctx_ladder": ctx_list,
                        "ready_seconds": ready_seconds,
                        "loaded_gpu": gpus_loaded,
                        "effective_ctx": ctx,
                        "served_ids": served_ids,
                        "server_log": log_path,
                    },
                })
                save_state()
            cleanup_old_logs()
            return True

        log(f"ctx={ctx} failed/timeout; killing and trying next")
        kill_llama_process()

    # All failed
    with state_lock:
        state.update({
            "state": "error", "pid": None, "updated_at": time.time(),
            "error": f"Failed to load {model_id} with any context size: {ctx_list}",
        })
        save_state()
    cleanup_old_logs()
    return False


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass
    
    def send_json(self, data, status=200):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
    
    def do_GET(self):
        if self.path == "/status":
            with state_lock:
                self.send_json(state)
        elif self.path == "/models":
            models = []
            for mid, spec in MODEL_CATALOG.items():
                models.append({
                    "id": mid,
                    "alias": spec["alias"],
                    "label": spec["label"],
                    "cuda": spec["cuda"],
                    "ctx": spec["ctx"],
                    "file": spec["file"],
                    "available": find_model_file(mid) is not None,
                })
            self.send_json({"models": models})
        else:
            self.send_json({"error": "not found"}, 404)
    
    def do_POST(self):
        if self.path == "/switch":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode()
                data = json.loads(body)
            except Exception:
                self.send_json({"error": "invalid json"}, 400)
                return
            
            if data.get("api_key") != API_KEY:
                self.send_json({"error": "unauthorized"}, 401)
                return
            
            model_id = data.get("model_id")
            if not model_id:
                self.send_json({"error": "missing model_id"}, 400)
                return
            
            # Run switch in background thread
            def run():
                with switch_lock:
                    do_switch(model_id, data.get("ctx"))
            
            threading.Thread(target=run, daemon=True).start()
            self.send_json({"status": "switching", "model_id": model_id})
        else:
            self.send_json({"error": "not found"}, 404)


def restore_runtime_state():
    """Re-sync in-memory state with reality after a service restart.

    If llama is actually serving, report ready (with its loaded alias) instead of
    leaving stale state (which is how /status ended up 'idle' while the state
    file claimed 'ready', and vice versa).
    """
    if check_llama_health():
        loaded = None
        try:
            req = urllib.request.Request("http://127.0.0.1:1234/v1/models", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            for m in data.get("data", []):
                loaded = m.get("id")
                break
        except Exception:
            pass
        # Map alias back to catalog id when possible
        model_id = None
        alias = loaded
        for mid, spec in MODEL_CATALOG.items():
            if loaded in (spec.get("alias"), mid):
                model_id = mid
                alias = spec.get("alias")
                break
        pids = find_llama_pids()
        with state_lock:
            state["state"] = "ready"
            state["model_id"] = model_id
            state["alias"] = alias
            state["pid"] = pids[0] if pids else None
            state["error"] = None
            state["updated_at"] = time.time()
        save_state()
        log(f"Restore: llama server found running, model={model_id or alias}, pid={state['pid']}")
    else:
        with state_lock:
            state["state"] = "idle"
            state["model_id"] = None
            state["alias"] = None
            state["pid"] = None
            state["updated_at"] = time.time()
        save_state()
        log("Restore: no llama server running, state=idle")


class ThreadedHTTPServer(http.server.ThreadingHTTPServer):
    daemon_threads = True


def main():
    log("Local LLM Manager starting")
    log(f"Model roots: {MODEL_ROOTS}")
    cleanup_old_logs()
    
    # Check available models
    for mid, spec in MODEL_CATALOG.items():
        path = find_model_file(mid)
        if path:
            log(f"  [OK] {spec['alias']}: {path}")
        else:
            log(f"  [MISSING] {spec['alias']}: {spec['file']} NOT FOUND")
    
    restore_runtime_state()
    
    server = ThreadedHTTPServer(("0.0.0.0", PORT), Handler)
    log(f"HTTP server listening on port {PORT}")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
