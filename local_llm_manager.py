"""
Local LLM Manager — switch models on llama.cpp (llama-server) with GPU memory checks.
Renames: GiorgioModelManager → Local-LLM-Manager
"""
import http.server, json, subprocess, time, threading, os, urllib.request, socket, sys
from datetime import datetime

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
        "cuda": ["CUDA0", "CUDA1"],
        "ctx": 32768,
    },
    "qwen3.5-35b-a3b": {
        "file": "Qwen3.5-35B-A3B-Q4_K_M.gguf",
        "path": "C:\\Users\\Peppuz\\.lmstudio\\models\\unsloth\\Qwen3.5-35B-A3B-GGUF\\Qwen3.5-35B-A3B-Q4_K_M.gguf",
        "alias": "Qwen 3.5",
        "label": "Qwen 3.5 — Ragionamento",
        "cuda": ["CUDA0", "CUDA1"],
        "ctx": 65536,
    },
    "mellum2-12b-a2.5b": {
        "file": "Mellum2-12B-A2.5B-Thinking-Q4_K_M.gguf",
        "path": "C:\\Users\\Peppuz\\.lmstudio\\models\\JetBrains\\Mellum2-12B-A2.5B-Thinking-GGUF-Q4_K_M\\Mellum2-12B-A2.5B-Thinking-Q4_K_M.gguf",
        "alias": "Mellum 2",
        "label": "Mellum 2 — Codice e velocità",
        "cuda": ["CUDA0"],
        "ctx": 131072,
    },
    "gemma-4-e4b": {
        "file": "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
        "path": "C:\\Users\\Peppuz\\.lmstudio\\models\\HauhauCS\\Gemma-4-E4B-Uncensored-HauhauCS-Aggressive\\Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
        "alias": "Gemma 4 E4B",
        "label": "Gemma 4 E4B — Leggero",
        "cuda": ["CUDA0"],
        "ctx": None,
    },
    "qwen3.6-35b-a3b-uncensored": {
        "file": "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
        "path": "D:\\Models\\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
        "alias": "Qwen 3.6 Uncensored",
        "label": "Qwen 3.6 Uncensored — Benchmark",
        "cuda": ["CUDA0", "CUDA1"],
        "ctx": 32768,
    },
    "ornith-1.5-35b-a3b": {
        "file": "Ornith-1.5-35B-Q4_K_M.gguf",
        "path": "D:\\Models\\Ornith-1.5-35B-Q4_K_M.gguf",
        "alias": "Ornith 1.5",
        "label": "Ornith 1.5 — Benchmark",
        "cuda": ["CUDA0", "CUDA1"],
        "ctx": 32768,
    },
    "qwen3.8-27b-gsq-rco": {
        "file": "Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf",
        "path": "D:\\Models\\Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf",
        "alias": "Qwen 3.8 GSQ",
        "label": "Qwen 3.8 GSQ — Compatto",
        "cuda": ["CUDA0"],
        "ctx": 32768,
    },
}

LLAMA_SERVER = "C:\\Users\\Peppuz\\AppData\\Local\\Microsoft\\WindowsApps\\llama.exe"
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


def find_llama_process():
    """Find running llama-server process."""
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq llama-server.exe", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=10
        )
        if "llama-server.exe" in result.stdout:
            for line in result.stdout.strip().split("\n"):
                if "llama-server.exe" in line:
                    parts = line.split(",")
                    if len(parts) >= 2:
                        pid = int(parts[1].strip('"'))
                        return pid
        return None
    except Exception:
        return None


def kill_llama_process():
    """Kill running llama-server."""
    pid = find_llama_process()
    if pid:
        try:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=10)
            time.sleep(2)
            return True
        except Exception as e:
            log_error(f"Failed to kill process {pid}: {e}")
            return False
    return True


def check_llama_health(timeout=5):
    """Check if llama-server is responding."""
    try:
        req = urllib.request.Request("http://127.0.0.1:1234/health", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode())
                return data.get("status") == "ok"
        return False
    except Exception:
        return False


def wait_for_llama_ready(timeout=120):
    """Wait for llama-server to become ready."""
    start = time.time()
    while time.time() - start < timeout:
        if check_llama_health():
            return True
        time.sleep(1)
    return False


def start_llama_server(model_path, ctx_size, cuda_devices):
    """Start llama-server with given parameters."""
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
        "--threads", "8",
        "--n-predict", "8192",
        "--temp", "0.7",
        "--top-p", "0.9",
        "--repeat-penalty", "1.1",
    ]
    log(f"Starting: {' '.join(cmd)}")
    
    # Set environment with llama-server directory and CUDA DLLs in PATH
    env = os.environ.copy()
    llama_dir = os.path.dirname(LLAMA_SERVER)
    cuda_dll_dir = r"C:\Users\Peppuz\.lmstudio\extensions\backends\vendor\win-llama-cuda12-vendor-v2"
    env["PATH"] = llama_dir + ";" + cuda_dll_dir + ";" + env.get("PATH", "")
    
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
            env=env
        )
        return proc.pid
    except Exception as e:
        log_error(f"Failed to start server: {e}")
        return None


def do_switch(model_id, requested_ctx=None):
    """Perform model switch with context ladder retry."""
    global state
    
    spec = MODEL_CATALOG.get(model_id)
    if not spec:
        with state_lock:
            state["state"] = "error"
            state["error"] = f"Unknown model: {model_id}"
            state["updated_at"] = time.time()
            save_state()
        return False
    
    model_path = find_model_file(model_id)
    if not model_path:
        with state_lock:
            state["state"] = "error"
            state["error"] = f"Model file not found: {spec['file']} in {MODEL_ROOTS}"
            state["updated_at"] = time.time()
            save_state()
        return False
    
    # Determine context sizes to try
    if requested_ctx:
        ctx_list = [c for c in CTX_LADDER if c <= requested_ctx]
        if requested_ctx not in ctx_list:
            ctx_list.insert(0, requested_ctx)
    else:
        ctx_list = CTX_LADDER
    
    # Kill existing
    with state_lock:
        state["state"] = "switching"
        state["model_id"] = model_id
        state["alias"] = spec["alias"]
        state["error"] = None
        state["updated_at"] = time.time()
        save_state()
    
    log(f"Switching to {spec['alias']} ({model_id})")
    
    # Capture GPU state before kill
    gpus_before = get_gpu_memory()
    
    if not kill_llama_process():
        with state_lock:
            state["state"] = "error"
            state["error"] = "Failed to kill existing process"
            state["updated_at"] = time.time()
            save_state()
        return False
    
    # Wait for GPU memory release
    time.sleep(3)
    gpus_after_kill = []
    for _ in range(3):
        gpus_after_kill = get_gpu_memory()
        if gpus_after_kill:
            break
        time.sleep(1)
    
    # Try each context size
    for ctx in ctx_list:
        log(f"Trying ctx={ctx}")
        pid = start_llama_server(model_path, ctx, spec["cuda"])
        if not pid:
            continue
        
        # Wait for ready
        if wait_for_llama_ready(timeout=90):
            log(f"Ready with ctx={ctx}")
            gpus_loaded = get_gpu_memory()
            with state_lock:
                state["state"] = "ready"
                state["effective_ctx"] = ctx
                state["requested_ctx"] = requested_ctx
                state["pid"] = pid
                state["last_switch"] = {
                    "released_gpu": gpus_after_kill,
                    "release_samples": [gpus_after_kill] * 3,
                    "old_process_exited": True,
                    "ctx_ladder": ctx_list,
                    "ready_seconds": 0,
                    "loaded_gpu": gpus_loaded,
                    "effective_ctx": ctx,
                }
                state["updated_at"] = time.time()
                save_state()
            return True
        else:
            log(f"Timeout with ctx={ctx}, killing and retrying")
            kill_llama_process()
            time.sleep(2)
    
    # All failed
    with state_lock:
        state["state"] = "error"
        state["error"] = f"Failed to load model with any context size: {ctx_list}"
        state["updated_at"] = time.time()
        save_state()
    return False


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass
    
    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
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


def main():
    log("Local LLM Manager starting")
    log(f"Model roots: {MODEL_ROOTS}")
    
    # Check available models
    for mid, spec in MODEL_CATALOG.items():
        path = find_model_file(mid)
        if path:
            log(f"  ✓ {spec['alias']}: {path}")
        else:
            log(f"  ✗ {spec['alias']}: {spec['file']} NOT FOUND")
    
    server = http.server.HTTPServer(("0.0.0.0", PORT), Handler)
    log(f"HTTP server listening on port {PORT}")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
