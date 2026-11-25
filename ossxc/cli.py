"""Interactive CLI client for OSSXC (direct local mode or connected to an OSSXC server via --url)."""
from __future__ import annotations

import argparse
import http.client
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ossxc.config import HF_MODEL_DIR, HF_MODEL_REPO, WEIGHTS_DIR


def _handle_common_command(user_text: str, messages: list[dict], state: dict) -> tuple[bool, bool]:
    if user_text in ("/quit", "/exit"):
        return True, True
    if user_text == "/reset":
        del messages[1:]
        print("[session reset]")
        return True, False
    if user_text == "/undo":
        if len(messages) >= 3 and messages[-1]["role"] == "assistant":
            del messages[-2:]
            print(f"[branched back to {len(messages)-1} messages — Ring-128 snapshot will restore on next turn]")
        else:
            print("[nothing to undo]")
        return True, False
    if user_text.startswith("/temp "):
        try:
            state["temperature"] = float(user_text.split(maxsplit=1)[1])
            mode = "greedy + N-gram" if state["temperature"] <= 0 else "sampling + prefetch"
            print(f"[temperature = {state['temperature']} ({mode})]")
        except ValueError:
            print("[usage: /temp 0.6]")
        return True, False
    if user_text.startswith("/top_p "):
        try:
            state["top_p"] = float(user_text.split(maxsplit=1)[1])
            print(f"[top_p = {state['top_p']}]")
        except ValueError:
            print("[usage: /top_p 0.9]")
        return True, False
    if user_text.startswith("/top_k "):
        try:
            state["top_k"] = max(0, int(user_text.split(maxsplit=1)[1]))
            print(f"[top_k = {state['top_k']}]")
        except ValueError:
            print("[usage: /top_k 40]")
        return True, False
    if user_text.startswith("/max "):
        try:
            state["max_tokens"] = int(user_text.split(maxsplit=1)[1])
            print(f"[max_tokens = {state['max_tokens']}]")
        except ValueError:
            print("[usage: /max 1024]")
        return True, False
    if user_text.startswith("/reasoning ") or user_text.startswith("/effort "):
        val = user_text.split(maxsplit=1)[1].strip().lower()
        if val in ("low", "medium", "high"):
            state["effort"] = val
            print(f"[reasoning_effort = {val}]")
        else:
            print("[usage: /reasoning low|medium|high]")
        return True, False
    return False, False


def run_cli_direct(args: argparse.Namespace) -> None:
    from ossxc.harmony import render_harmony_prompt_ids
    from ossxc.runtime import PersistentOSSXCRuntime

    use_hybrid = (not args.pure_fp16_kv) and (args.ctx > 4096)
    print(f"Loading OSSXC directly (ctx={args.ctx}, hybrid_kv={use_hybrid}, temp={args.temp})...")
    rt = PersistentOSSXCRuntime(
        model_dir=args.model_dir,
        weights_dir=args.weights_dir,
        kv_cap_global=args.ctx,
        hybrid_kv=use_hybrid,
    )
    mem = rt.engine.get_mem_info()
    print(f"Ready | VRAM={mem['total_engine_vram_gb']:.3f} GB ({mem['nvidia_smi_used_mib']:.0f} MiB) | "
          f"C_VRAM={mem['cache_capacity_slots']} slots | RSS={mem['process_ram_rss_gb']:.2f} GB")
    print("Commands: /reset, /undo, /stats, /temp <v>, /top_p <v>, /top_k <n>, /max <n>, /reasoning <low|medium|high>, /quit\n")

    state = {"temperature": args.temp, "top_p": args.top_p, "top_k": args.top_k, "max_tokens": args.max_tokens, "effort": args.effort}
    messages = [{"role": "system", "content": "You are a concise, expert software engineering assistant."}]

    try:
        while True:
            try:
                user_text = input("\nYou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_text:
                continue
            if user_text == "/stats":
                m = rt.engine.get_mem_info()
                print(f"[KV L={len(rt.kv_tokens)}, snapshots={len(rt.snapshots)}, "
                      f"VRAM={m['total_engine_vram_gb']:.3f} GB, smi={m['nvidia_smi_used_mib']:.0f} MiB, "
                      f"RSS={m['process_ram_rss_gb']:.2f} GB]")
                continue
            handled, quit_now = _handle_common_command(user_text, messages, state)
            if quit_now:
                break
            if handled:
                continue

            messages.append({"role": "user", "content": user_text})
            prompt_ids, _ = render_harmony_prompt_ids(rt.tokenizer, messages, reasoning_effort=state["effort"])
            in_analysis = in_final = False
            final_text = ""
            cancel_ev = threading.Event()
            try:
                for ev in rt.generate_stream(
                    prompt_tokens=prompt_ids,
                    max_tokens=state["max_tokens"],
                    temperature=state["temperature"],
                    top_p=state["top_p"],
                    top_k=state["top_k"],
                    cancel_event=cancel_ev,
                ):
                    if ev["type"] == "meta":
                        c = ev["continuity"]
                        print(f"[KV {c.case}: reused={c.reused_prefix_tokens}, new={c.new_suffix_tokens}, TTFT={c.ttft_ms:.1f} ms]")
                    elif ev["type"] == "delta":
                        d = ev["delta"]
                        if d.reasoning_content and args.show_thinking:
                            if not in_analysis:
                                sys.stdout.write("\n[analysis] ")
                                in_analysis = True
                            sys.stdout.write(d.reasoning_content)
                            sys.stdout.flush()
                        if d.content:
                            if not in_final:
                                sys.stdout.write("\nAssistant> ")
                                in_final = True
                            sys.stdout.write(d.content)
                            sys.stdout.flush()
                            final_text += d.content
                    elif ev["type"] == "done":
                        print(f"\n[done: {ev['completion_tokens']} toks @ {ev['gen_tps']:.2f} tok/s | KV L={ev['kv_L']}]")
                        final_text = ev["content"]
            except KeyboardInterrupt:
                cancel_ev.set()
                print("\n[cancelled]")
            messages.append({"role": "assistant", "content": final_text})
    finally:
        rt.close()


def run_cli_http(args: argparse.Namespace) -> None:
    base = args.url.rstrip("/")
    endpoint = base if base.endswith("/v1/chat/completions") else (base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions")
    state = {"temperature": args.temp, "top_p": args.top_p, "top_k": args.top_k, "max_tokens": args.max_tokens, "effort": args.effort}
    messages = [{"role": "system", "content": "You are a concise, expert software engineering assistant."}]
    print(f"Connected to {endpoint} (temp={state['temperature']}, top_p={state['top_p']}, top_k={state['top_k']}, effort={state['effort']}).")
    print("Commands: /reset, /undo, /temp <v>, /top_p <v>, /top_k <n>, /max <n>, /reasoning <low|medium|high>, /quit\n")

    while True:
        try:
            user_text = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        handled, quit_now = _handle_common_command(user_text, messages, state)
        if quit_now:
            break
        if handled:
            continue

        messages.append({"role": "user", "content": user_text})
        body = json.dumps({
            "model": HF_MODEL_REPO,
            "messages": messages,
            "stream": True,
            "temperature": state["temperature"],
            "top_p": state["top_p"],
            "top_k": state["top_k"],
            "max_tokens": state["max_tokens"],
            "reasoning_effort": state["effort"],
        }).encode("utf-8")
        req = urllib.request.Request(endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST")

        in_analysis = in_final = False
        final_parts: list[str] = []
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    obj = json.loads(data_str)
                    if obj.get("choices"):
                        d = obj["choices"][0].get("delta", {})
                        if d.get("reasoning_content") and args.show_thinking:
                            if not in_analysis:
                                sys.stdout.write("\n[analysis] ")
                                in_analysis = True
                            sys.stdout.write(d["reasoning_content"])
                            sys.stdout.flush()
                        if d.get("content"):
                            if not in_final:
                                sys.stdout.write("\nAssistant> ")
                                in_final = True
                            sys.stdout.write(d["content"])
                            sys.stdout.flush()
                            final_parts.append(d["content"])
                    if obj.get("x_perf"):
                        xp = obj["x_perf"]
                        xc = obj.get("x_continuity") or {}
                        print(f"\n[done: {xp.get('gen_tps', 0):.2f} tok/s | KV {xc.get('case', '')} "
                              f"reused={xc.get('reused_prefix_tokens', 0)} TTFT={xc.get('ttft_ms', 0):.1f} ms]")
        except (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError, OSError) as exc:
            print(f"\n[connection error] {exc}")
            messages.pop()
            continue
        except KeyboardInterrupt:
            print("\n[cancelled]")
        messages.append({"role": "assistant", "content": "".join(final_parts)})


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive CLI for OSSXC (GPT-OSS-20B MXFP4)")
    ap.add_argument("--url", default=None, help="Connect to running OSSXC server (e.g. http://127.0.0.1:8000)")
    ap.add_argument("--ctx", type=int, default=65536, help="Direct mode context length (default: 65536)")
    ap.add_argument("--model-dir", default=HF_MODEL_DIR)
    ap.add_argument("--weights-dir", default=WEIGHTS_DIR)
    ap.add_argument("--pure-fp16-kv", action="store_true")
    ap.add_argument("--temp", type=float, default=0.0, help="Sampling temperature (0.0 = greedy + N-gram)")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0 = disabled)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--effort", choices=["low", "medium", "high"], default="medium")
    ap.add_argument("--show-thinking", action="store_true", help="Print Harmony analysis channel")
    args = ap.parse_args()

    if args.url:
        run_cli_http(args)
    else:
        run_cli_direct(args)


if __name__ == "__main__":
    main()
