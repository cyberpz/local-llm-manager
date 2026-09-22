"""
Local LLM Manager v3 — request-driven model loader with queue.
Transparent OpenAI-compatible proxy on port 1235, forwards to llama-server on 1234.
No manual /switch endpoint — models load automatically based on request's model field.
"""
import http.server
import http.client
import json
import subprocess
import time
import threading
import os
import urllib.request
import urllib.error
import urllib.parse
import socket
import sys
import queue
from datetime import datetime

# Windows console may be cp1252: force UTF-8 so log lines never crash the process
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

MODEL_ROOTS = [
    "C:\\Users\\Peppuz",
    "D:\\Models",
]

MODEL_CATALOG = {
    "nemotron-cascade-2-30b-a3b": {
        "file": "nvidia_Nemotron-Cascade-2-30B-A3B-Q4_0.gguf",
        "path": "D:\\Models\\lmstudio\\bartowski\\nvidia_Nemotron-Cascade-2-30B-A3B-GGUF\\nvidia_Nemotron-Cascade-2-30B-A3B-Q4_0.gguf",
        "alias": "Nemotron Cascade 2",
        "label": "Nemotron Cascade 2 — Qualità",
        "cuda": ["Vulkan0", "Vulkan1"],
        "ctx": 131072,
    },
    "mellum2-12b-a2.5b": {
        "file": "Mellum2-12B-A2.5B-Thinking-Q4_K_M.gguf",
        "path": "D:\\Models\\lmstudio\\JetBrains\\Mellum2-12B-A2.5B-Thinking-GGUF-Q4_K_M\\Mellum2-12B-A2.5B-Thinking-Q4_K_M.gguf",
        "alias": "Mellum 2",
        "label": "Mellum 2 — Codice e velocità",
        "cuda": ["Vulkan0", "Vulkan1"],
        "ctx": 131072,
    },
    "gemma-4-e4b": {
        "file": "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
        "path": "D:\\Models\\lmstudio\\HauhauCS\\Gemma-4-E4B-Uncensored-HauhauCS-Aggressive\\Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
        "alias": "Gemma 4 E4B",
        "label": "Gemma 4 E4B — Leggero",
        "cuda": ["Vulkan0"],
        "ctx": 131072,
    },
    "qwen3.6-35b-a3b-uncensored": {
        "file": "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
        "path": "D:\\Models\\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
        "alias": "Qwen 3.6 Uncensored",
        "label": "Qwen 3.6 Uncensored — Benchmark",
        "cuda": ["Vulkan0", "Vulkan1"],
        "ctx": 131072,
    },
    "ornith-1.5-35b-a3b": {
        "file": "Ornith-1.5-35B-Q4_K_M.gguf",
        "path": "D:\\Models\\Ornith-1.5-35B-Q4_K_M.gguf",
        "alias": "Ornith 1.5",
        "label": "Ornith 1.5 — Benchmark",
        "cuda": ["Vulkan0", "Vulkan1"],
        "ctx": 131072,
    },
    "qwen3.8-27b-gsq-rco": {
        "file": "Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf",
        "path": "D:\\Models\\Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf",
        "alias": "Qwen 3.8 GSQ",
        "label": "Qwen 3.8 GSQ — Compatto",
        "cuda": ["Vulkan0", "Vulkan1"],
        "ctx": 64001,
    },
    "minicpm-2b-q8": {
        "file": "MiniCPM5-2B-heretic-abliterated-Q8_0.gguf",
        "path": "D:\\Models\\MiniCPM5-2B-heretic-abliterated-Q8_0.gguf",
        "alias": "MiniCPM Q8",
        "label": "MiniCPM Q8",
        "cuda": ["Vulkan0"],
        "ctx": 131072,
    },
    "Spark-X2.5-4B-Q8": {
        "file": "Spark-X2.5-4B-Q8_0.gguf",
        "path": "D:\\Models\\Spark-X2.5-4B-Q8_0.gguf",
        "alias": "Spark-X2.5-4B-Q8_0",
        "label": "Spark-X2.5-4B-Q8_0",
        "cuda": ["Vulkan0", "Vulkan1"],
        "ctx": 131072,
    },
}

LLAMA_SERVER = r"C:\Users\Peppuz\AppData\Local\Microsoft\WindowsApps\llama.exe"
BASE_DIR = r"C:\Users\Peppuz"
STATE_FILE = os.path.join(BASE_DIR, "local-llm-manager-state.json")
LOG_FILE = os.path.join(BASE_DIR, "local-llm-manager.log")
ERROR_LOG_FILE = os.path.join(BASE_DIR, "local-llm-manager-error.log")
API_KEY = "giorgio-local-manager"
PROXY_PORT = 1235
LLAMA_PORT = 1234
HEALTH_INTERVAL = 10  # seconds between health checks

CTX_LADDER = [512000, 256000, 131072, 65536, 32768, 16384, 8192]

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
    "queue_size": 0,
}
state_lock = threading.Lock()

# Request queue for model loading
request_queue = queue.Queue()
switch_lock = threading.Lock()  # Prevents concurrent model switches


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
    print(line, file=sys.stderr, flush=True)
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


def load_state():
    """Load persisted state from disk. Returns dict or None."""
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def find_model_file(model_id):
    spec = MODEL_CATALOG.get(model_id)
    if not spec:
        return None
    if "path" in spec and os.path.isfile(spec["path"]):
        return spec["path"]
    filename = spec["file"]
    for root in MODEL_ROOTS:
        path = os.path.join(root, filename)
        if os.path.isfile(path):
            return path
    return None


def get_gpu_memory():
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


def port_llama_busy():
    try:
        with socket.create_connection(("127.0.0.1", LLAMA_PORT), timeout=1):
            return True
    except OSError:
        return False


def kill_llama_process():
    pids = find_llama_pids()
    for pid in pids:
        try:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=10)
        except Exception as e:
            log_error(f"Failed to kill process {pid}: {e}")
    if pids:
        time.sleep(2)
    for _ in range(10):
        if not port_llama_busy():
            return True
        time.sleep(1)
    if port_llama_busy():
        log_error(f"Port {LLAMA_PORT} still busy after kill attempts")
        return False
    return True


def check_llama_health(timeout=5, expected_alias=None):
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{LLAMA_PORT}/health", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return False
        if expected_alias:
            try:
                req2 = urllib.request.Request(f"http://127.0.0.1:{LLAMA_PORT}/v1/models", method="GET")
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
    if not pid:
        return False
    try:
        r = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return f'"{pid}"' in (r.stdout or "")
    except Exception:
        return True


def wait_for_llama_ready(timeout=180, pid=None, expected_alias=None):
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
    cuda_str = ",".join(cuda_devices)
    cmd = [
        LLAMA_SERVER,
        "serve",
        "-m", model_path,
        "-c", str(ctx_size),
        "--host", "0.0.0.0",
        "--port", str(LLAMA_PORT),
        "-ngl", "999",
        "--device", cuda_str,
        "--threads", "4",
        "-fa", "on",
        "-np", "1",
        "--cache-type-k", "q4_0",
        "--cache-type-v", "q4_0",
    ]
    if alias:
        cmd += ["--alias", alias]
    log_path = os.path.join(BASE_DIR, f"llama-server-{int(time.time())}.log")
    log(f"Starting: {' '.join(cmd)} (log: {log_path})")

    env = os.environ.copy()
    llama_dir = os.path.dirname(LLAMA_SERVER)
    env["PATH"] = llama_dir + ";" + env.get("PATH", "")

    try:
        logf = open(log_path, "ab")
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
            env=env
        )
        # Close our copy of the file handle; the child owns it now
        logf.close()
        return proc.pid, log_path
    except Exception as e:
        log_error(f"Failed to start server: {e}")
        try:
            logf.close()
        except Exception:
            pass
        return None, None


def cleanup_old_logs(keep=5):
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
    catalog_ctx = spec.get("ctx")
    if requested_ctx:
        ladder = [c for c in CTX_LADDER if c <= requested_ctx]
        if requested_ctx not in ladder:
            ladder.insert(0, requested_ctx)
        return ladder or [requested_ctx]
    if catalog_ctx:
        # Use catalog ctx as max — never try values above what the catalog defines
        ladder = [c for c in CTX_LADDER if c <= catalog_ctx]
        if catalog_ctx not in ladder:
            ladder.insert(0, catalog_ctx)
        return ladder or [catalog_ctx]
    return list(CTX_LADDER)


def load_model(model_id, requested_ctx=None):
    """Load a model into llama-server. Returns True on success."""
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
            "state": "loading", "model_id": model_id, "alias": spec["alias"],
            "error": None, "updated_at": time.time(),
        })
        save_state()

    log(f"Loading {spec['alias']} ({model_id}), ctx ladder: {ctx_list}")

    old_pids = find_llama_pids()
    killed_ok = kill_llama_process()
    old_process_exited = killed_ok and not (set(old_pids) & set(find_llama_pids()))
    print(f'Killed {old_process_exited}')

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
                req = urllib.request.Request(f"http://127.0.0.1:{LLAMA_PORT}/v1/models", method="GET")
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

    with state_lock:
        state.update({
            "state": "error", "pid": None, "updated_at": time.time(),
            "error": f"Failed to load {model_id} with any context size: {ctx_list}",
        })
        save_state()
    cleanup_old_logs()
    return False


def forward_to_llama(method, path, headers, body, stream_callback=None):
    """Forward request to llama-server using http.client for proper streaming support.
    
    Args:
        method: HTTP method
        path: Request path
        headers: Request headers dict
        body: Request body bytes
        stream_callback: If provided, called with (status, headers_dict, chunk) for streaming
    
    Returns:
        dict with status, headers, body for non-streaming
        True/False for streaming (success/failure)
    """
    conn = None
    try:
        conn = http.client.HTTPConnection("127.0.0.1", LLAMA_PORT, timeout=300)
        
        # Prepare headers for forwarding
        forward_headers = {}
        for key, value in headers.items():
            key_lower = key.lower()
            if key_lower not in ('host', 'content-length', 'transfer-encoding'):
                forward_headers[key] = value
        
        conn.request(method, path, body=body, headers=forward_headers)
        response = conn.getresponse()
        
        if stream_callback:
            # Streaming mode
            stream_callback(response.status, dict(response.getheaders()), response)
            return True
        else:
            # Non-streaming mode
            response_body = response.read()
            return {
                "status": response.status,
                "headers": dict(response.getheaders()),
                "body": response_body,
            }
    except Exception as e:
        if stream_callback:
            # Send error via callback
            error_body = json.dumps({"error": f"Upstream error: {str(e)}"}).encode()
            stream_callback(502, {"Content-Type": "application/json"}, error_body)
            return True
        else:
            return {
                "status": 502,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": f"Upstream error: {str(e)}"}).encode(),
            }
    finally:
        # Don't close connection in streaming mode — the HTTP handler
        # reads from response_obj and closes it when done
        if conn and not stream_callback:
            try:
                conn.close()
            except Exception:
                pass


def request_worker():
    """Worker thread that processes requests from the queue."""
    while True:
        try:
            # Get next request from queue
            item = request_queue.get()
            if item is None:  # Shutdown signal
                break
            
            request_data, done_event, response_result, reading_done_event = item
            
            with state_lock:
                state["queue_size"] = request_queue.qsize()
                save_state()
            
            model_id = request_data.get("model_id")
            is_streaming = request_data.get("stream", False)
            method = request_data.get("method")
            path = request_data.get("path")
            headers = request_data.get("headers")
            body = request_data.get("body")
            
            # Check if we need to load a different model
            with state_lock:
                current_model = state.get("model_id")
                current_state = state.get("state")
            
            if current_model != model_id or current_state != "ready":
                # Need to load the model
                log(f"Request for {model_id}, currently loaded: {current_model}. Loading...")
                with switch_lock:
                    # Double-check after acquiring lock
                    with state_lock:
                        current_model = state.get("model_id")
                        current_state = state.get("state")
                    
                    if current_model != model_id or current_state != "ready":
                        success = load_model(model_id, request_data.get("ctx"))
                        if not success:
                            # Model failed to load
                            error_body = json.dumps({
                                "error": f"Failed to load model: {model_id}"
                            }).encode()
                            response_result["status"] = 503
                            response_result["headers"] = {"Content-Type": "application/json"}
                            response_result["body"] = error_body
                            response_result["streaming"] = False
                            done_event.set()
                            request_queue.task_done()
                            continue
            
            # Model is ready, forward the request
            if is_streaming:
                # Streaming response
                def stream_handler(status, resp_headers, response_obj):
                    response_result["status"] = status
                    response_result["headers"] = resp_headers
                    response_result["response_obj"] = response_obj
                    response_result["streaming"] = True
                    done_event.set()
                
                forward_to_llama(method, path, headers, body, stream_callback=stream_handler)
                
                # Wait for HTTP handler to finish reading the stream
                # This prevents killing llama-server while a stream is being read
                reading_done_event.wait()
            else:
                # Non-streaming response
                response = forward_to_llama(method, path, headers, body)
                response_result["status"] = response["status"]
                response_result["headers"] = response["headers"]
                response_result["body"] = response["body"]
                response_result["streaming"] = False
                done_event.set()
            
            request_queue.task_done()
            
        except Exception as e:
            log_error(f"Worker error: {e}")
            try:
                response_result["status"] = 500
                response_result["headers"] = {"Content-Type": "application/json"}
                response_result["body"] = json.dumps({"error": str(e)}).encode()
                response_result["streaming"] = False
                done_event.set()
                request_queue.task_done()
            except Exception:
                pass


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress default logging

    def send_json(self, data, status=200):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def check_auth(self):
        """Check API key from query param or Authorization header."""
        # Check query parameter
        if "?" in self.path:
            path_part, query_part = self.path.split("?", 1)
            params = urllib.parse.parse_qs(query_part)
            if params.get("api_key", [None])[0] == API_KEY:
                return True
        
        # Check Authorization header
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
            if token == API_KEY:
                return True
        
        return False

    def do_GET(self):
        path = self.path.split("?")[0]
        
        if path == "/health":
            self.send_json({"status": "ok"})
        elif path == "/status":
            with state_lock:
                self.send_json(state)
        elif path == "/v1/models":
            # Return all models from catalog
            models = []
            for mid, spec in MODEL_CATALOG.items():
                models.append({
                    "id": mid,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                    "alias": spec["alias"],
                    "label": spec["label"],
                    "cuda": spec["cuda"],
                    "ctx": spec["ctx"],
                    "file": spec["file"],
                    "available": find_model_file(mid) is not None,
                })
            self.send_json({"object": "list", "data": models})
        else:
            # Pass through to llama-server
            if not self.check_auth():
                self.send_json({"error": "unauthorized"}, 401)
                return
            
            response = forward_to_llama("GET", self.path, self.headers, None)
            self.send_response(response["status"])
            for key, value in response["headers"].items():
                if key.lower() not in ('transfer-encoding', 'connection'):
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(response["body"])))
            self.end_headers()
            self.wfile.write(response["body"])

    def do_POST(self):
        path = self.path.split("?")[0]
        
        if not self.check_auth():
            self.send_json({"error": "unauthorized"}, 401)
            return
        
        # Read request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""
        
        if path == "/v1/chat/completions":
            # Intercept and queue for model loading
            try:
                data = json.loads(body.decode()) if body else {}
            except Exception:
                self.send_json({"error": "invalid json"}, 400)
                return
            
            model_id = data.get("model")
            if not model_id:
                self.send_json({"error": "missing model field"}, 400)
                return
            
            if model_id not in MODEL_CATALOG:
                self.send_json({"error": f"Unknown model: {model_id}"}, 400)
                return
            
            is_streaming = data.get("stream", False)
            
            # Queue the request
            request_data = {
                "model_id": model_id,
                "stream": is_streaming,
                "method": "POST",
                "path": self.path,
                "headers": dict(self.headers),
                "body": body,
                "ctx": data.get("ctx"),
            }
            
            done_event = threading.Event()
            reading_done_event = threading.Event()
            response_result = {}
            
            with state_lock:
                state["queue_size"] = request_queue.qsize() + 1
                save_state()
            
            request_queue.put((request_data, done_event, response_result, reading_done_event))
            
            # Wait for worker to process and signal completion
            done_event.wait()
            
            # Send response
            if response_result.get("streaming"):
                # Streaming response - forward chunks from upstream
                self.send_response(response_result["status"])
                for key, value in response_result["headers"].items():
                    if key.lower() not in ('transfer-encoding', 'connection'):
                        self.send_header(key, value)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                
                # Stream response body from upstream
                response_obj = response_result["response_obj"]
                try:
                    while True:
                        chunk = response_obj.read(8192)
                        if not chunk:
                            break
                        # Write chunk in chunked encoding format
                        chunk_size = f"{len(chunk):X}\r\n".encode()
                        self.wfile.write(chunk_size)
                        self.wfile.write(chunk)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    
                    # Write final chunk
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                finally:
                    try:
                        # Close the underlying connection, not just the response
                        response_obj.close()
                    except Exception:
                        pass
                    # Signal worker that we're done reading
                    reading_done_event.set()
            else:
                # Non-streaming response
                self.send_response(response_result["status"])
                for key, value in response_result["headers"].items():
                    if key.lower() not in ('transfer-encoding', 'connection'):
                        self.send_header(key, value)
                self.send_header("Content-Length", str(len(response_result["body"])))
                self.end_headers()
                self.wfile.write(response_result["body"])
            
        else:
            # Pass through to llama-server
            response = forward_to_llama("POST", self.path, self.headers, body)
            self.send_response(response["status"])
            for key, value in response["headers"].items():
                if key.lower() not in ('transfer-encoding', 'connection'):
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(response["body"])))
            self.end_headers()
            self.wfile.write(response["body"])


def restore_runtime_state():
    """Re-sync state with reality. If llama is running, adopt it.
    If not but state file says ready, try to auto-restore last model."""
    if check_llama_health():
        loaded = None
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{LLAMA_PORT}/v1/models", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            for m in data.get("data", []):
                loaded = m.get("id")
                break
        except Exception:
            pass
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
        # No llama running. Try auto-restore from persisted state.
        saved = load_state()
        if saved and saved.get("state") == "ready" and saved.get("model_id"):
            last_model = saved["model_id"]
            saved_ctx = saved.get("effective_ctx")
            # Use catalog ctx as ceiling — never exceed what the catalog defines
            catalog_ctx = MODEL_CATALOG.get(last_model, {}).get("ctx")
            if catalog_ctx and saved_ctx:
                last_ctx = min(saved_ctx, catalog_ctx)
            else:
                last_ctx = catalog_ctx or saved_ctx
            log(f"Restore: no llama running but state says '{last_model}' was loaded. Auto-restoring...")
            with state_lock:
                state["state"] = "idle"
                state["updated_at"] = time.time()
            save_state()
            # Fire auto-restore in background so HTTP server starts immediately
            threading.Thread(
                target=_auto_restore,
                args=(last_model, last_ctx),
                daemon=True,
            ).start()
        else:
            with state_lock:
                state["state"] = "idle"
                state["model_id"] = None
                state["alias"] = None
                state["pid"] = None
                state["updated_at"] = time.time()
            save_state()
            log("Restore: no llama server running, state=idle")


def _auto_restore(model_id, ctx):
    """Background auto-restore of last loaded model after restart."""
    time.sleep(2)  # let HTTP server bind first
    with switch_lock:
        ok = load_model(model_id, ctx)
    if ok:
        log(f"Auto-restore: {model_id} loaded successfully")
    else:
        log(f"Auto-restore: FAILED to load {model_id}")


def health_monitor():
    """Periodic health check. Detects dead llama processes and updates state."""
    while True:
        time.sleep(HEALTH_INTERVAL)
        with state_lock:
            current_state = state["state"]
            current_pid = state["pid"]
            current_model = state["model_id"]

        if current_state != "ready":
            continue

        # Check if process is alive
        if current_pid and not proc_alive(current_pid):
            log(f"Health monitor: pid {current_pid} DEAD (model={current_model})")
            # Double-check with HTTP health
            if not check_llama_health():
                log(f"Health monitor: /health confirms server down. Setting state=error")
                with state_lock:
                    state["state"] = "error"
                    state["error"] = f"Server process {current_pid} died unexpectedly"
                    state["pid"] = None
                    state["updated_at"] = time.time()
                save_state()
            else:
                # Process died but another took over? Update PID
                new_pids = find_llama_pids()
                if new_pids:
                    log(f"Health monitor: different llama pid detected: {new_pids[0]}")
                    with state_lock:
                        state["pid"] = new_pids[0]
                        state["updated_at"] = time.time()
                    save_state()
            continue

        # Process alive, check HTTP health
        if not check_llama_health():
            log(f"Health monitor: pid {current_pid} alive but /health failed")
            with state_lock:
                state["state"] = "error"
                state["error"] = f"Server pid {current_pid} alive but /health unresponsive"
                state["updated_at"] = time.time()
            save_state()


class ThreadedHTTPServer(http.server.ThreadingHTTPServer):
    daemon_threads = True


def main():
    log("Local LLM Manager v3 starting")
    log(f"Model roots: {MODEL_ROOTS}")
    cleanup_old_logs()

    for mid, spec in MODEL_CATALOG.items():
        path = find_model_file(mid)
        if path:
            log(f"  [OK] {spec['alias']}: {path}")
        else:
            log(f"  [MISSING] {spec['alias']}: {spec['file']} NOT FOUND")

    restore_runtime_state()

    # Start request worker thread
    worker_thread = threading.Thread(target=request_worker, daemon=True)
    worker_thread.start()
    log("Request worker thread started")

    # Start health monitor thread
    monitor_thread = threading.Thread(target=health_monitor, daemon=True)
    monitor_thread.start()
    log(f"Health monitor started (interval={HEALTH_INTERVAL}s)")

    server = ThreadedHTTPServer(("0.0.0.0", PROXY_PORT), ProxyHandler)
    log(f"Proxy server listening on port {PROXY_PORT}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("Shutting down")
        request_queue.put(None)  # Shutdown signal to worker
        server.shutdown()


if __name__ == "__main__":
    main()
