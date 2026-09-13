#!/usr/bin/env python3
"""BenchChat server: static files (no-cache) + /api/fleet aggregation.

Replaces `python3 -m http.server`. Serves the tools/ directory and exposes
/api/fleet, which queries every AINode node API for loaded models and returns
ready-to-use OpenAI endpoint URLs, so the BenchChat dropdown reflects reality
instead of hardcoded labels. Extra non-AINode endpoints (e.g. the GLM pair
launched outside AINode) are listed in EXTRA below until they move into the
AINode launch path.
"""
import base64
import hmac
import json
import os
import threading
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from concurrent.futures import ThreadPoolExecutor

NODES = {
    "Spark-1-DGX": "100.122.26.9",
    "Spark-2-DGX": "100.81.184.19",
    "Spark-3-DGX": "100.80.240.119",
    "Spark-4-GX10": "100.72.9.84",
    "c4130-ai-01": "100.112.28.89",
}
STATUS_HOSTS = list(dict.fromkeys(NODES.values()))
EXTRA = [
    {"label": "GLM-5.3-Flash TP2 DFlash2 @ spark-2/3 :8020", "url": "http://100.81.184.19:8020/v1",
     "meta": {"nodes": ["Spark-2-DGX", "Spark-3-DGX"], "tp": 2, "spec": "DFlash2"}},
    {"label": "DeepSeek-V4-Flash DSpark 1M :8888", "url": "http://100.81.184.19:8888/v1",
     "meta": {"nodes": ["Spark-2-DGX"], "tp": 1, "engine": "DSpark"}},
    {"label": "GLM-5.3 UD-Q3 llama.cpp @ m3-studio :8940", "url": "http://100.84.108.16:8940/v1",
     "meta": {"nodes": ["m3-studio"], "tp": 1, "engine": "llama.cpp", "quant": "UD-Q3",
              "repo": "unsloth/GLM-5.3-GGUF"}},
    {"label": "Qwen3.8-27B UD-Q4 llama.cpp @ m3-studio :8941", "url": "http://100.84.108.16:8941/v1",
     "meta": {"nodes": ["m3-studio"], "tp": 1, "engine": "llama.cpp", "quant": "UD-Q4",
              "repo": "unsloth/Qwen3.8-27B-GGUF"}},
    {"label": "Qwen3.8-Flash-Next NVFP4 @ c4130 4xV100 TP4 MTP4 :8104", "url": "http://100.112.28.89:8104/v1",
     "meta": {"nodes": ["c4130-ai-01"], "tp": 4, "quant": "NVFP4", "spec": "MTP-4",
              "engine": "1Cat-vLLM (src build)"}},
    {"label": "Qwen3.8-27B NVFP4 @ c4130 4xV100 TP4 128K :8101", "url": "http://100.112.28.89:8101/v1",
     "meta": {"nodes": ["c4130-ai-01"], "tp": 4, "quant": "NVFP4",
              "engine": "1Cat-vLLM (src build)", "repo": "unsloth/Qwen3.8-27B-NVFP4"}},
]

# Hardware facts per host, for the model card. AINode nodes report gpu_name and
# gpu_memory_gb over /api/cluster/info, so only non-AINode boxes are hardcoded.
# c4130 topology confirmed by `nvidia-smi topo -m`: PHB inside each NUMA pair,
# SYS across, no NVLink.
HOSTS = {
    "100.112.28.89": {"node": "c4130-ai-01", "gpu": "Tesla V100 32GB", "gpus": 4, "vram_gb": 32.0,
                      "link": "PCIe (PHB in-pair / SYS across, no NVLink)"},
    "100.84.108.16": {"node": "m3-studio", "gpu": "Apple M3 Ultra (unified)", "gpus": 1, "vram_gb": 512.0,
                      "link": "on-package"},
}

# Upstream repo for each served model, so the card can link out. Keys match the
# model id AINode reports, or the EXTRA label for engines launched outside it.
REPOS = {
    "unsloth/Qwen3.8-27B-NVFP4": "unsloth/Qwen3.8-27B-NVFP4",
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4":
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3.8-flash-next": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
    "glm-5.3-flash": "zai-org/GLM-5.3-Flash",
}


# 1x1 PNG, the smallest thing that makes an engine decide whether it accepts an
# image at all.
_PIXEL = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
          "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
_CAPS_CACHE = {}   # (hostport, model) -> caps dict


def _post(hostport, body, timeout):
    req = urllib.request.Request(
        f"http://{hostport}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}
    except Exception:
        return None, {}


def probe_caps(hostport, model):
    """Ask the engine what it will actually accept, rather than inferring it
    from the model name. Each probe is a real request capped at a token or two.
    A capability can be present in the weights and still off here (we serve
    Flash-Next with --language-model-only), which is the distinction the card
    needs to show.

    Reasoning is deliberately not probed. These models choose per prompt whether
    to think, so a single request cannot separate "won't" from "didn't" -- the
    client marks it from turns actually observed instead."""
    key = (hostport, model)
    if key in _CAPS_CACHE:
        return _CAPS_CACHE[key]

    caps = {"vision": None, "tools": None}

    code, body = _post(hostport, {
        "model": model, "max_tokens": 1,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": _PIXEL}}]}]}, 45)
    if code is not None:
        msg = str(body.get("error", {}).get("message", "")).lower()
        caps["vision"] = code == 200
        if code != 200:
            caps["vision_why"] = ("served text-only" if "at most 0 image" in msg
                                  else "not supported")

    code, body = _post(hostport, {
        "model": model, "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {
            "name": "ping", "description": "p",
            "parameters": {"type": "object", "properties": {}}}}]}, 30)
    if code is not None:
        msg = str(body.get("error", {}).get("message", "")).lower()
        caps["tools"] = code == 200
        if code != 200:
            caps["tools_why"] = ("no --tool-call-parser" if "tool-call-parser" in msg
                                 else "not supported")

    _CAPS_CACHE[key] = caps
    return caps


def hf_tags(repo):
    """pipeline_tag says what the weights can do, independent of how we serve."""
    if not repo:
        return {}
    try:
        with urllib.request.urlopen(
                f"https://huggingface.co/api/models/{repo}", timeout=8) as r:
            d = json.load(r)
        return {"pipeline_tag": d.get("pipeline_tag"),
                "model_vision": d.get("pipeline_tag") in
                                ("image-text-to-text", "visual-question-answering",
                                 "image-to-text", "any-to-any")}
    except Exception:
        return {}


def fetch_status(host):
    try:
        with urllib.request.urlopen(f"http://{host}:3000/api/server/status", timeout=4) as r:
            return json.load(r).get("loaded_models", [])
    except Exception:
        return []


def fetch_cluster(host):
    try:
        with urllib.request.urlopen(f"http://{host}:3000/api/cluster/info", timeout=4) as r:
            return {m["node_name"]: m for m in json.load(r).get("members", [])}
    except Exception:
        return {}


def hw_for(nodes, members):
    """Describe the silicon behind an endpoint. HOSTS wins where it has an
    entry, because AINode's cluster view reports one GPU per node and the c4130
    carries four. Sparks fall through to their live gpu_name/gpu_memory_gb."""
    for h in HOSTS.values():
        if h["node"] in nodes:
            return {"gpu": h["gpu"], "gpus_per_node": h["gpus"],
                    "vram_gb": h.get("vram_gb"), "link": h.get("link")}
    for n in nodes:
        mem = members.get(n)
        if mem:
            return {"gpu": mem.get("gpu_name"), "vram_gb": mem.get("gpu_memory_gb"),
                    "gpus_per_node": 1}
    return {}


def fleet():
    seen, out = set(), []
    with ThreadPoolExecutor(max_workers=8) as ex:
        statuses = list(ex.map(fetch_status, STATUS_HOSTS))
        members = {}
        for c in ex.map(fetch_cluster, STATUS_HOSTS):
            members.update(c)
    for models in statuses:
        for m in models:
            ip = NODES.get(m.get("node_hostname", ""), "")
            key = (m.get("id"), m.get("node_hostname"), m.get("port"))
            if not ip or key in seen:
                continue
            seen.add(key)
            short = m["id"].split("/")[-1]
            node = m["node_hostname"]
            repo = REPOS.get(m["id"])
            meta = {"model_id": m["id"], "nodes": [node], "tp": m.get("parallel") or 1,
                    "port": m.get("port"), "managed": "AINode",
                    "quant": m.get("quantization"), "format": m.get("format"),
                    "capabilities": m.get("capabilities") or [],
                    "loaded_at": m.get("loaded_at"),
                    "repo": repo, "hf": f"https://huggingface.co/{repo}" if repo else None}
            meta.update(hw_for([node], members))
            out.append({
                "label": f"{short} @ {node.replace('-DGX','').replace('-GX10','').lower()} "
                         f":{m['port']}" + ("" if m.get("ready") else " (loading)"),
                "url": f"/proxy/{ip}:{m['port']}/v1",
                "ready": bool(m.get("ready")),
                "meta": meta,
            })

    extra = []
    for e in EXTRA:
        hostport = e["url"].split("//", 1)[1].split("/", 1)[0]
        meta = dict(e.get("meta") or {})
        meta.setdefault("managed", "raw docker (outside AINode)")
        meta["port"] = int(hostport.rsplit(":", 1)[1])
        repo = meta.get("repo")
        if not repo:
            for k, v in REPOS.items():
                if k.split("/")[-1].lower() in e["label"].lower():
                    repo = v
                    break
        meta["repo"] = repo
        meta["hf"] = f"https://huggingface.co/{repo}" if repo else None
        meta.update(hw_for(meta.get("nodes") or [], members))
        extra.append({"label": e["label"], "url": "/proxy/" + e["url"].split("//", 1)[1],
                      "meta": meta})
    return {"endpoints": out + extra}


# Basic auth so this can sit on a public hostname. Unset password = open, which
# is what you want on a laptop and never what you want behind a domain.
BENCHCHAT_USER = os.environ.get("BENCHCHAT_USER", "bench")
BENCHCHAT_PASSWORD = os.environ.get("BENCHCHAT_PASSWORD", "")


def _authorized(header):
    if not BENCHCHAT_PASSWORD:
        return True
    if not header or not header.startswith("Basic "):
        return False
    try:
        user, _, pw = base64.b64decode(header[6:]).decode().partition(":")
    except Exception:
        return False
    # compare_digest on both halves so neither leaks length by timing
    return (hmac.compare_digest(user, BENCHCHAT_USER)
            and hmac.compare_digest(pw, BENCHCHAT_PASSWORD))



class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, must-revalidate")
        super().end_headers()

    def _deny(self):
        body = b"authentication required"
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="BenchChat"')
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not _authorized(self.headers.get("Authorization")):
            self._deny()
            return
        if self.path.rstrip("/") == "/api/fleet":
            body = json.dumps(fleet()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/api/caps?"):
            q = urllib.parse.parse_qs(self.path.split("?", 1)[1])
            ep, model = q.get("ep", [""])[0], q.get("model", [""])[0]
            if ep not in ALLOWED_PROXY:
                self.send_error(403, "target not in fleet")
                return
            if q.get("fresh", [""])[0]:
                _CAPS_CACHE.pop((ep, model), None)   # engine relaunched: re-probe
            out = dict(probe_caps(ep, model))
            out.update(hf_tags(q.get("repo", [""])[0]))
            body = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/proxy/"):
            self._proxy(method="GET")
            return
        super().do_GET()

    def do_POST(self):
        if not _authorized(self.headers.get("Authorization")):
            self._deny()
            return
        # /proxy/<host:port>/<rest> forwards to an engine so the browser only
        # ever talks to this origin. Removes the CORS and routing failures that
        # happen when a client can reach BenchChat but not a node directly.
        if self.path.startswith("/proxy/"):
            self._proxy()
            return
        self.send_error(404)

    def _proxy(self, method="POST"):
        target = self.path[len("/proxy/"):]
        if "/" not in target:
            self.send_error(400, "expected /proxy/host:port/path")
            return
        hostport, rest = target.split("/", 1)
        if hostport not in ALLOWED_PROXY:
            self.send_error(403, "target not in fleet")
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = self.rfile.read(length) if length else None
        req = urllib.request.Request(
            f"http://{hostport}/{rest}", data=payload, method=method,
            headers={"Content-Type": self.headers.get("Content-Type", "application/json"),
                     "Accept": self.headers.get("Accept", "*/*")},
        )
        # Stream the upstream response through chunk by chunk. Buffering it would
        # collapse TTFT and decode-rate into one number and make every timing in
        # BenchChat meaningless.
        try:
            r = urllib.request.urlopen(req, timeout=1800)
        except urllib.error.HTTPError as e:
            body = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        except Exception as e:
            body = json.dumps({"error": str(e)}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        with r:
            ctype = r.headers.get("Content-Type", "application/json")
            self.send_response(r.status)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            # A cold long prefill can go minutes before the first token. Behind a
            # CDN that reads as a dead origin and the request gets cut (524), so
            # trickle SSE comments until real output starts. Comments are part of
            # the event-stream spec and every client ignores them.
            sse = "text/event-stream" in ctype
            first, lock = threading.Event(), threading.Lock()

            def keepalive():
                while not first.wait(15):
                    with lock:
                        try:
                            self.wfile.write(b": waiting on prefill\n\n")
                            self.wfile.flush()
                        except Exception:
                            return

            if sse:
                threading.Thread(target=keepalive, daemon=True).start()
            try:
                while True:
                    chunk = r.read1(8192) if hasattr(r, "read1") else r.read(8192)
                    if not chunk:
                        break
                    first.set()
                    with lock:
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                first.set()

    def log_message(self, *a):
        pass


def _allowed():
    hosts = {f"{ip}:{p}" for ip in NODES.values() for p in (3000, 8000, 8001, 8020, 8101, 8102, 8104)}
    for e in EXTRA:
        hosts.add(e["url"].split("//", 1)[1].split("/", 1)[0])
    return hosts


ALLOWED_PROXY = _allowed()

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    host = os.environ.get("BENCHCHAT_HOST", "0.0.0.0")
    port = int(os.environ.get("BENCHCHAT_PORT", "8899"))
    guard = "basic auth on" if BENCHCHAT_PASSWORD else "NO AUTH (set BENCHCHAT_PASSWORD)"
    print(f"BenchChat on {host}:{port} — static + /api/fleet + /proxy — {guard}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
