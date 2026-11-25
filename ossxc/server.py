"""OSSXC OpenAI-Compatible Local HTTP/SSE Server — Hardened for OpenCode & Agentic Clients.
Endpoints:
  - GET  /v1/models  (and /models)
  - POST /v1/chat/completions  (and /chat/completions)
  - GET  /health
  - GET  /v1/stats

Supports OpenCode (@ai-sdk/openai-compatible) out of the box:
  - Streaming SSE with separated delta + finish_reason chunks + stream_options usage + prompt_tokens_details.cached_tokens
  - Harmony reasoning_content, function tool_calls, multi-tool-call history normalization, and tool_choice="none"
  - Safe handling of null temperature/top_p/max_tokens, CORS preflight, and live client-disconnect cancellation
"""
from __future__ import annotations

import argparse
import json
import select
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ossxc.config import HF_MODEL_DIR, HF_MODEL_REPO, WEIGHTS_DIR
from ossxc.engine_wrapper import CACHE_ADAPTIVE_HYBRID
from ossxc.harmony import render_harmony_prompt_ids
from ossxc.runtime import RING128_KV_DATA_BYTES, PersistentOSSXCRuntime

MODEL_ID = HF_MODEL_REPO


def _is_client_disconnected(sock: socket.socket) -> bool:
    try:
        r, _, x = select.select([sock], [], [sock], 0.0)
        if x:
            return True
        if r:
            data = sock.recv(1, socket.MSG_PEEK)
            if len(data) == 0:
                return True
        return False
    except BlockingIOError:
        return False
    except OSError:
        return True


class OSSXCServerState:
    def __init__(self, runtime: PersistentOSSXCRuntime, model_id: str = MODEL_ID):
        self.runtime = runtime
        self.model_id = model_id
        self.start_time = int(time.time())
        self.last_continuity: dict[str, Any] | None = None
        self.last_finish_reason: str | None = None
        self.last_completion_tokens: int = 0
        self.request_count = 0


def _make_handler(state: OSSXCServerState):
    class OpenAIHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args):
            sys.stderr.write(f"[ossxc-server] {self.address_string()} - {fmt % args}\n")

        def _send_cors_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Title, HTTP-Referer")

        def _send_json(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass

        def do_OPTIONS(self):
            self.send_response(204)
            self._send_cors_headers()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?", 1)[0].rstrip("/")
            if path in ("/v1/models", "/models"):
                self._send_json(200, {
                    "object": "list",
                    "data": [{
                        "id": state.model_id,
                        "object": "model",
                        "created": state.start_time,
                        "owned_by": "ossxc-mxfp4-local",
                        "max_context": state.runtime.kv_cap_global,
                    }],
                })
                return
            if path in ("/health", "/v1/health", ""):
                self._send_json(200, {"status": "ok", "engine": "OSSXC", "model": state.model_id})
                return
            if path in ("/v1/stats", "/stats"):
                with state.runtime.lock:
                    mem = state.runtime.engine.get_mem_info()
                    snaps = [
                        {
                            "abs_pos_S": s.abs_pos,
                            "valid_len": s.valid_len,
                            "kv_data_bytes": s.kv_data_bytes,
                            "metadata_bytes": s.metadata_bytes,
                            "sha256_prefix": s.prefix_sha256[:16],
                        }
                        for s in state.runtime.snapshots.values()
                    ]
                    kv_L = len(state.runtime.kv_tokens)
                self._send_json(200, {
                    "engine": "OSSXC",
                    "model": state.model_id,
                    "requests_served": state.request_count,
                    "kv_processed_L": kv_L,
                    "ring128_snapshot_kv_bytes": RING128_KV_DATA_BYTES,
                    "snapshots_count": len(snaps),
                    "snapshots": snaps,
                    "last_continuity": state.last_continuity,
                    "last_finish_reason": state.last_finish_reason,
                    "last_completion_tokens": state.last_completion_tokens,
                    "memory": mem,
                })
                return
            self._send_json(404, {"error": {"message": f"Unknown endpoint {path}", "type": "invalid_request_error"}})

        def do_POST(self):
            path = self.path.split("?", 1)[0].rstrip("/")
            if path not in ("/v1/chat/completions", "/chat/completions"):
                self._send_json(404, {"error": {"message": f"Unknown endpoint {path}", "type": "invalid_request_error"}})
                return

            try:
                clen = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(clen) if clen > 0 else b"{}"
                req = json.loads(raw.decode("utf-8"))
            except Exception as exc:
                self._send_json(400, {"error": {"message": f"Invalid JSON body: {exc}", "type": "invalid_request_error"}})
                return

            messages = req.get("messages")
            if not isinstance(messages, list) or not messages:
                self._send_json(400, {"error": {"message": "'messages' must be a non-empty list", "type": "invalid_request_error"}})
                return

            req_model = str(req.get("model") or state.model_id)
            tool_choice = req.get("tool_choice")
            tools = None if tool_choice == "none" else req.get("tools")
            stream = bool(req.get("stream", False))

            raw_max = req.get("max_completion_tokens")
            if raw_max is None:
                raw_max = req.get("max_tokens")
            max_tokens = 2048 if raw_max is None else max(1, int(raw_max))

            raw_temp = req.get("temperature")
            temperature = 0.0 if raw_temp is None else max(0.0, float(raw_temp))
            raw_top_p = req.get("top_p")
            top_p = 1.0 if raw_top_p is None else min(1.0, max(1e-4, float(raw_top_p)))

            raw_effort = (
                req.get("reasoning_effort")
                or (req.get("reasoning", {}).get("effort") if isinstance(req.get("reasoning"), dict) else None)
            )
            if raw_effort is None:
                # Auto-use concise reasoning for short-budget requests (e.g. OpenCode title generation with max_tokens<=256)
                reasoning_effort = "low" if max_tokens <= 256 else "medium"
            else:
                reasoning_effort = str(raw_effort).lower()
            if reasoning_effort not in ("low", "medium", "high"):
                reasoning_effort = "medium"

            seed = req.get("seed")
            stop_raw = req.get("stop")
            if isinstance(stop_raw, str):
                stop_strings = [stop_raw]
            elif isinstance(stop_raw, list):
                stop_strings = [str(x) for x in stop_raw if x]
            else:
                stop_strings = []

            try:
                prompt_ids, _ = render_harmony_prompt_ids(
                    state.runtime.tokenizer,
                    messages=messages,
                    tools=tools,
                    reasoning_effort=reasoning_effort,
                )
            except Exception as exc:
                self._send_json(400, {"error": {"message": f"Harmony prompt rendering error: {exc}", "type": "invalid_request_error"}})
                return

            avail = state.runtime.kv_cap_global - len(prompt_ids) - 2
            if avail < 1:
                self._send_json(400, {
                    "error": {
                        "message": f"Prompt tokens ({len(prompt_ids)}) exceed context limit ({state.runtime.kv_cap_global})",
                        "type": "invalid_request_error",
                    }
                })
                return
            max_tokens = min(max_tokens, avail)

            req_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
            created = int(time.time())
            cancel_ev = threading.Event()
            state.request_count += 1

            if stream:
                self._handle_stream(
                    req_id, created, req_model, prompt_ids, max_tokens, temperature, top_p, stop_strings, seed, cancel_ev
                )
            else:
                self._handle_non_stream(
                    req_id, created, req_model, prompt_ids, max_tokens, temperature, top_p, stop_strings, seed, cancel_ev
                )

        def _write_sse(self, data_str: str) -> bool:
            chunk = f"data: {data_str}\n\n".encode("utf-8")
            try:
                self.wfile.write(chunk)
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                return False

        def _handle_stream(
            self,
            req_id: str,
            created: int,
            req_model: str,
            prompt_ids: list[int],
            max_tokens: int,
            temperature: float,
            top_p: float,
            stop_strings: list[str],
            seed: int | None,
            cancel_ev: threading.Event,
        ) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self._send_cors_headers()
            self.end_headers()
            self.close_connection = True

            init_chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req_model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            if not self._write_sse(json.dumps(init_chunk, ensure_ascii=False)):
                cancel_ev.set()
                return

            tool_idx = 0
            cont_dict: dict[str, Any] | None = None
            emitted_finish = False
            stream_done = threading.Event()

            def _disconnect_watcher():
                while not stream_done.is_set():
                    if _is_client_disconnected(self.connection):
                        cancel_ev.set()
                        return
                    stream_done.wait(0.025)

            watcher = threading.Thread(target=_disconnect_watcher, daemon=True)
            watcher.start()

            try:
                for ev in state.runtime.generate_stream(
                    prompt_tokens=prompt_ids,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop_strings=stop_strings,
                    cancel_event=cancel_ev,
                    seed=seed,
                ):
                    if not cancel_ev.is_set() and _is_client_disconnected(self.connection):
                        cancel_ev.set()

                    if ev["type"] == "meta":
                        c = ev["continuity"]
                        cont_dict = c.__dict__.copy()
                        state.last_continuity = cont_dict
                    elif ev["type"] == "delta":
                        if cancel_ev.is_set():
                            continue
                        d = ev["delta"]
                        delta_obj: dict[str, Any] = {}
                        if d.reasoning_content:
                            delta_obj["reasoning_content"] = d.reasoning_content
                        if d.content:
                            delta_obj["content"] = d.content
                        if d.tool_call:
                            delta_obj["tool_calls"] = [{
                                "index": tool_idx,
                                "id": d.tool_call["id"],
                                "type": "function",
                                "function": d.tool_call["function"],
                            }]
                            tool_idx += 1

                        # OpenCode / @ai-sdk/openai-compatible: emit content/tool_calls first with finish_reason=None,
                        # then emit a clean terminal chunk with delta={} and finish_reason!
                        if delta_obj:
                            payload = {
                                "id": req_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": req_model,
                                "choices": [{"index": 0, "delta": delta_obj, "finish_reason": None}],
                            }
                            if not self._write_sse(json.dumps(payload, ensure_ascii=False)):
                                cancel_ev.set()
                                continue
                        if d.finish_reason:
                            fin_payload = {
                                "id": req_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": req_model,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": d.finish_reason}],
                            }
                            emitted_finish = True
                            if not self._write_sse(json.dumps(fin_payload, ensure_ascii=False)):
                                cancel_ev.set()
                    elif ev["type"] == "done":
                        fin = ev.get("finish_reason", "stop")
                        state.last_finish_reason = fin
                        state.last_completion_tokens = int(ev.get("completion_tokens", 0))
                        if not cancel_ev.is_set():
                            if not emitted_finish:
                                fin_payload = {
                                    "id": req_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": req_model,
                                    "choices": [{"index": 0, "delta": {}, "finish_reason": fin}],
                                }
                                self._write_sse(json.dumps(fin_payload, ensure_ascii=False))
                            cached_toks = int(cont_dict.get("reused_prefix_tokens", 0)) if cont_dict else 0
                            final_chunk = {
                                "id": req_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": req_model,
                                "choices": [],
                                "usage": {
                                    "prompt_tokens": len(prompt_ids),
                                    "completion_tokens": ev["completion_tokens"],
                                    "total_tokens": len(prompt_ids) + ev["completion_tokens"],
                                    "prompt_tokens_details": {"cached_tokens": cached_toks},
                                },
                                "x_continuity": cont_dict,
                                "x_perf": {
                                    "gen_tps": ev["gen_tps"],
                                    "gen_seconds": ev["gen_seconds"],
                                    "kv_L": ev["kv_L"],
                                },
                            }
                            self._write_sse(json.dumps(final_chunk, ensure_ascii=False))
                            self._write_sse("[DONE]")
            finally:
                stream_done.set()
                watcher.join(timeout=0.2)

        def _handle_non_stream(
            self,
            req_id: str,
            created: int,
            req_model: str,
            prompt_ids: list[int],
            max_tokens: int,
            temperature: float,
            top_p: float,
            stop_strings: list[str],
            seed: int | None,
            cancel_ev: threading.Event,
        ) -> None:
            cont_dict: dict[str, Any] | None = None
            done_ev: dict[str, Any] = {}
            for ev in state.runtime.generate_stream(
                prompt_tokens=prompt_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_strings=stop_strings,
                cancel_event=cancel_ev,
                seed=seed,
            ):
                if ev["type"] == "meta":
                    cont_dict = ev["continuity"].__dict__.copy()
                    state.last_continuity = cont_dict
                elif ev["type"] == "done":
                    done_ev = ev
                    state.last_finish_reason = ev.get("finish_reason")
                    state.last_completion_tokens = int(ev.get("completion_tokens", 0))

            msg: dict[str, Any] = {
                "role": "assistant",
                "content": done_ev.get("content") if not done_ev.get("tool_calls") else (done_ev.get("content") or None),
            }
            if done_ev.get("reasoning_content"):
                msg["reasoning_content"] = done_ev["reasoning_content"]
            if done_ev.get("tool_calls"):
                msg["tool_calls"] = done_ev["tool_calls"]

            cached_toks = int(cont_dict.get("reused_prefix_tokens", 0)) if cont_dict else 0
            resp = {
                "id": req_id,
                "object": "chat.completion",
                "created": created,
                "model": req_model,
                "choices": [{
                    "index": 0,
                    "message": msg,
                    "finish_reason": done_ev.get("finish_reason", "stop"),
                }],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": done_ev.get("completion_tokens", 0),
                    "total_tokens": len(prompt_ids) + done_ev.get("completion_tokens", 0),
                    "prompt_tokens_details": {"cached_tokens": cached_toks},
                },
                "x_continuity": cont_dict,
                "x_perf": {
                    "gen_tps": done_ev.get("gen_tps", 0.0),
                    "gen_seconds": done_ev.get("gen_seconds", 0.0),
                    "kv_L": done_ev.get("kv_L", 0),
                },
            }
            self._send_json(200, resp)

    return OpenAIHandler


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    model_dir: str = HF_MODEL_DIR,
    weights_dir: str = WEIGHTS_DIR,
    kv_cap_global: int = 65536,
    enable_ngram_spec: bool = True,
    hybrid_kv: bool | None = None,
    lookahead_prefetch: bool = True,
    max_vram_ceiling_bytes: int | None = None,
) -> tuple[ThreadingHTTPServer, OSSXCServerState]:
    rt = PersistentOSSXCRuntime(
        model_dir=model_dir,
        weights_dir=weights_dir,
        kv_cap_global=kv_cap_global,
        cache_policy=CACHE_ADAPTIVE_HYBRID,
        num_copy_threads=2,
        enable_ngram_spec=enable_ngram_spec,
        verify_readonly_cache=True,
        hybrid_kv=hybrid_kv,
        lookahead_prefetch=lookahead_prefetch,
        max_vram_ceiling_bytes=max_vram_ceiling_bytes,
    )
    state = OSSXCServerState(rt)
    server = ThreadingHTTPServer((host, port), _make_handler(state))
    server.daemon_threads = True
    return server, state


def main() -> None:
    ap = argparse.ArgumentParser(description="OSSXC Local OpenAI/OpenCode-Compatible Server for GPT-OSS-20B (MXFP4)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ctx", type=int, default=65536, help="Max global KV context length (4096..65536, default: 65536)")
    ap.add_argument("--model-dir", default=HF_MODEL_DIR, help="Path to HF tokenizer/chat_template directory")
    ap.add_argument("--weights-dir", default=WEIGHTS_DIR, help="Path to packed OSSXC weights directory")
    ap.add_argument("--ceiling-gb", type=float, default=None, help="Optional VRAM arena ceiling in decimal GB (e.g. 5.19 for 6GB GPUs, 7.5 for 8GB GPUs)")
    ap.add_argument("--pure-fp16-kv", action="store_true", help="Force Pure FP16 KV even when --ctx > 4096")
    ap.add_argument("--no-ngram", action="store_true", help="Disable greedy N-gram speculation")
    ap.add_argument("--no-prefetch", action="store_true", help="Disable Inter-Layer Lookahead DMA Prefetch")
    args = ap.parse_args()

    use_hybrid = (not args.pure_fp16_kv) and (args.ctx > 4096)
    ceiling_bytes = int(args.ceiling_gb * 1e9) if args.ceiling_gb is not None else None
    kv_label = "HYBRID K-FP16 + V_pc32 (exact boundary)" if use_hybrid else "Pure FP16"
    pref_label = "OFF" if args.no_prefetch else "ON (M3_ema, 0-copy swap, tau=0.04)"

    print(f"Loading OSSXC — {MODEL_ID} (MXFP4 native 4-bit, KV={args.ctx} [{kv_label}], prefetch={pref_label}, ngram={not args.no_ngram})...")
    server, state = create_server(
        host=args.host,
        port=args.port,
        model_dir=args.model_dir,
        weights_dir=args.weights_dir,
        kv_cap_global=args.ctx,
        enable_ngram_spec=not args.no_ngram,
        hybrid_kv=use_hybrid,
        lookahead_prefetch=not args.no_prefetch,
        max_vram_ceiling_bytes=ceiling_bytes,
    )
    mem = state.runtime.engine.get_mem_info()
    lm_ms = state.runtime.engine.benchmark_lmhead(iters=10, use_fast=True)
    print(
        f"OSSXC ready on http://{args.host}:{args.port}/v1 (OpenAI & OpenCode compatible)\n"
        f"  VRAM={mem['total_engine_vram_gb']:.3f} GB ({mem['nvidia_smi_used_mib']:.0f} MiB nvidia-smi) | "
        f"C_VRAM={mem['cache_capacity_slots']} slots | lm_head={lm_ms:.2f} ms | RSS={mem['process_ram_rss_gb']:.2f} GB"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down OSSXC server...")
    finally:
        server.server_close()
        state.runtime.close()


if __name__ == "__main__":
    main()
