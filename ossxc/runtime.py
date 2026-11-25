"""Persistent OSSXC Runtime with Exact 4-Case KV Prefix Continuity & Ring-128 Snapshots.
Implements:
  - Single persistent openai/gpt-oss-20b MXFP4 engine across multi-turn agentic sessions
  - Exact 4-case KV prefix reuse (A: direct extension, B: exact Ring-128 snapshot, C: rollback+reprocess, D: cold prefill)
  - Auto Hybrid KV (exact K-FP16 + V_hot4K + V_sink128 + Cold V_pc32 + RAM mirror + V_bnd_fp16 boundary parity) for ctx > 4K
  - Inter-Layer Speculative DMA Lookahead Prefetch (M3_ema + 1-ns zero-copy pointer swap)
  - O(1) hash-indexed N-gram lookup speculation (K=6, min_n=5, greedy mode only) with read-only cache & bit-exact rollback
  - Exact temperature + top_p sampling when temperature > 0.0 (with lookahead prefetch still active!)
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from .config import HF_MODEL_DIR, WEIGHTS_DIR
from .engine_wrapper import CACHE_ADAPTIVE_HYBRID, OSSXCEngine
from .harmony import (
    HARMONY_STOP_TOKEN_IDS,
    TOK_CALL,
    HarmonyStreamParser,
    load_official_tokenizer,
)

RING128_KV_DATA_BYTES = 12 * 2 * 128 * 8 * 64 * 2  # exact 3,145,728 B = 3 MiB


@dataclass
class Ring128Snapshot:
    k_ring: np.ndarray            # [12, 128, 512] uint16
    v_ring: np.ndarray            # [12, 128, 512] uint16
    abs_pos: int
    ring_cursors: tuple[int, ...]
    valid_len: int
    prefix_tokens: tuple[int, ...]
    prefix_sha256: str

    @property
    def kv_data_bytes(self) -> int:
        return int(self.k_ring.nbytes + self.v_ring.nbytes)

    @property
    def metadata_bytes(self) -> int:
        return len(self.prefix_tokens) * 4 + len(self.ring_cursors) * 4 + len(self.prefix_sha256) + 24


@dataclass
class ContinuityReport:
    case: str
    prev_L: int
    prompt_len: int
    common_prefix_P: int
    restored_snapshot_S: int | None
    reused_prefix_tokens: int
    reprocessed_tokens_S_to_P: int
    new_suffix_tokens: int
    total_tokens_processed: int
    ring_restore_ms: float
    prefill_ms: float
    ttft_ms: float
    hot_kv_restore_ms: float = 0.0


class _FastNgramIndex:
    """O(1) hash-indexed prompt/history N-gram drafter (zero slowdown even at 64K context)."""

    def __init__(self, tokens: list[int], min_n: int = 5, max_n: int = 6):
        self.min_n = min_n
        self.max_n = max_n
        self.tokens: list[int] = []
        self.tables: dict[int, dict[tuple[int, ...], int]] = {n: {} for n in range(min_n, max_n + 1)}
        self.extend(tokens)

    def extend(self, new_tokens: list[int]) -> None:
        for t in new_tokens:
            self.tokens.append(int(t))
            L = len(self.tokens)
            for n in range(self.min_n, self.max_n + 1):
                if L >= n + 1:
                    key = tuple(self.tokens[L - n - 1 : L - 1])
                    self.tables[n][key] = L - 1

    def draft(self, tail: list[int], K: int) -> list[int]:
        if K <= 0:
            return []
        L = len(self.tokens)
        for n in range(self.max_n, self.min_n - 1, -1):
            if len(tail) < n:
                continue
            pos = self.tables[n].get(tuple(tail[-n:]))
            if pos is not None and pos < L:
                cand = self.tokens[pos : min(L, pos + K)]
                if len(cand) >= 2:
                    return cand
        return []


def _hash_tokens(tokens: list[int] | tuple[int, ...]) -> str:
    arr = np.asarray(tokens, dtype=np.int32)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def sample_token(
    logits: np.ndarray,
    temperature: float = 0.0,
    top_p: float = 1.0,
    rng: np.random.Generator | None = None,
) -> int:
    if temperature <= 0.0:
        return int(np.argmax(logits))
    scaled = logits.astype(np.float64) / float(temperature)
    scaled -= np.max(scaled)
    probs = np.exp(scaled)
    probs /= np.sum(probs)
    if 0.0 < top_p < 1.0:
        order = np.argsort(probs)[::-1]
        sorted_p = probs[order]
        csum = np.cumsum(sorted_p)
        cutoff = int(np.searchsorted(csum, top_p)) + 1
        keep_idx = order[:cutoff]
        keep_p = probs[keep_idx]
        keep_p /= np.sum(keep_p)
        g = rng if rng is not None else np.random.default_rng()
        return int(g.choice(keep_idx, p=keep_p))
    g = rng if rng is not None else np.random.default_rng()
    return int(g.choice(len(probs), p=probs))


class PersistentOSSXCRuntime:
    """Thread-safe persistent runtime owning the single GPU engine, Harmony tokenizer,
    exact KV continuity state, and Ring-128 RAM snapshot store."""

    def __init__(
        self,
        model_dir: str = HF_MODEL_DIR,
        weights_dir: str = WEIGHTS_DIR,
        kv_cap_global: int = 65536,
        cache_policy: int = CACHE_ADAPTIVE_HYBRID,
        num_copy_threads: int = 2,
        chunk_size: int = 512,
        max_snapshots: int = 64,
        enable_ngram_spec: bool = True,
        ngram_k: int = 6,
        verify_readonly_cache: bool = True,
        hybrid_kv: bool | None = None,
        lookahead_prefetch: bool = True,
        lookahead_tau: float = 0.040,
        max_vram_ceiling_bytes: int | None = None,
    ):
        self.model_dir = model_dir
        self.weights_dir = weights_dir
        self.kv_cap_global = kv_cap_global
        self.cache_policy = cache_policy
        self.chunk_size = chunk_size
        self.max_snapshots = max_snapshots
        self.enable_ngram_spec = enable_ngram_spec
        self.ngram_k = ngram_k
        self.verify_readonly_cache = verify_readonly_cache
        self.hybrid_kv = (kv_cap_global > 4096) if hybrid_kv is None else bool(hybrid_kv)
        self.lookahead_prefetch = bool(lookahead_prefetch)
        self.lookahead_tau = float(lookahead_tau)

        if max_vram_ceiling_bytes is None:
            vram_ceiling = 5_190_000_000 if kv_cap_global >= 32768 else 5_150_000_000
        else:
            vram_ceiling = int(max_vram_ceiling_bytes)
        safety_margin = (32 * 1024 * 1024) if kv_cap_global >= 32768 else (48 * 1024 * 1024)

        self.lock = threading.Lock()
        self.tokenizer = load_official_tokenizer(model_dir)
        self.engine = OSSXCEngine(
            weights_dir=weights_dir,
            kv_cap_global=kv_cap_global,
            max_vram_ceiling_bytes=vram_ceiling,
            device_safety_margin_bytes=safety_margin,
            cache_policy=cache_policy,
            num_copy_threads=num_copy_threads,
            hybrid_kv=self.hybrid_kv,
            pinned_experts=True,
        )
        self.engine.set_verify_readonly_cache(verify_readonly_cache)
        self.engine.set_quest_prune_margin(14.0, collect_stats=False)
        self.engine.set_lookahead_prefetch(1 if self.lookahead_prefetch else 0, self.lookahead_tau)
        self.engine.set_profile_enabled(False)

        self.kv_tokens: list[int] = []
        self.last_logits: np.ndarray | None = None
        self.snapshots: OrderedDict[tuple[int, str], Ring128Snapshot] = OrderedDict()

    def save_ring_snapshot(self) -> Ring128Snapshot:
        S = len(self.kv_tokens)
        k_ring, v_ring = self.engine.read_ring_kv()
        pref_tuple = tuple(self.kv_tokens)
        sha = _hash_tokens(pref_tuple)
        snap = Ring128Snapshot(
            k_ring=k_ring,
            v_ring=v_ring,
            abs_pos=S,
            ring_cursors=tuple((S % 128) for _ in range(12)),
            valid_len=min(S, 128),
            prefix_tokens=pref_tuple,
            prefix_sha256=sha,
        )
        key = (S, sha)
        if key in self.snapshots:
            self.snapshots.move_to_end(key)
        self.snapshots[key] = snap
        while len(self.snapshots) > self.max_snapshots:
            self.snapshots.popitem(last=False)
        return snap

    def _find_compatible_snapshot(self, new_tokens: list[int], max_pos: int) -> Ring128Snapshot | None:
        best: Ring128Snapshot | None = None
        for snap in self.snapshots.values():
            S = snap.abs_pos
            if S > max_pos or S > len(self.kv_tokens):
                continue
            if best is not None and S <= best.abs_pos:
                continue
            if snap.prefix_tokens == tuple(new_tokens[:S]) and self.kv_tokens[:S] == new_tokens[:S]:
                best = snap
        return best

    def _process_token_span(self, tokens: list[int], start_pos: int, compute_last_logits: bool = True) -> np.ndarray | None:
        n = len(tokens)
        if n == 0:
            return self.last_logits
        if start_pos + n > self.kv_cap_global:
            raise ValueError(f"Prompt length {start_pos + n} exceeds KV capacity {self.kv_cap_global}")

        logits = None
        if n <= 8:
            for i, tid in enumerate(tokens):
                logits, _ = self.engine.forward_token(int(tid), start_pos + i, return_top4=False)
        else:
            logits = self.engine.prefill_sequence(tokens, chunk_size=self.chunk_size, start_pos=start_pos)
            self.engine.set_cache_policy(self.cache_policy)
        self.kv_tokens.extend(int(t) for t in tokens)
        if compute_last_logits:
            self.last_logits = logits
        return logits

    def prepare_prompt(self, new_tokens: list[int], save_boundary_snapshot: bool = True) -> tuple[np.ndarray, ContinuityReport]:
        t0 = time.perf_counter()
        if len(new_tokens) == 0:
            raise ValueError("Empty prompt token list")
        if len(new_tokens) > self.kv_cap_global:
            raise ValueError(f"Prompt length {len(new_tokens)} exceeds KV capacity {self.kv_cap_global}")

        prev_L = len(self.kv_tokens)
        P = _common_prefix_len(self.kv_tokens, new_tokens)

        if P == len(new_tokens) and P == prev_L and self.last_logits is not None:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            rep = ContinuityReport(
                case="A_DIRECT", prev_L=prev_L, prompt_len=len(new_tokens), common_prefix_P=P,
                restored_snapshot_S=None, reused_prefix_tokens=P, reprocessed_tokens_S_to_P=0,
                new_suffix_tokens=0, total_tokens_processed=0, ring_restore_ms=0.0,
                prefill_ms=0.0, ttft_ms=round(dt_ms, 3),
            )
            return self.last_logits.copy(), rep

        target_p = P if P < len(new_tokens) else max(0, P - 1)
        ring_restore_ms = 0.0
        hot_kv_restore_ms = 0.0

        if prev_L > 0 and target_p == prev_L:
            case = "A_DIRECT"
            restored_S = None
            reprocessed = 0
            suffix = new_tokens[prev_L:]
            t_p0 = time.perf_counter()
            logits = self._process_token_span(suffix, start_pos=prev_L, compute_last_logits=True)
            prefill_ms = (time.perf_counter() - t_p0) * 1000.0
        else:
            snap = self._find_compatible_snapshot(new_tokens, target_p) if target_p > 0 else None
            if snap is not None and snap.abs_pos == target_p:
                case = "B_EXACT_SNAP"
                restored_S = snap.abs_pos
                t_r0 = time.perf_counter()
                self.engine.write_ring_kv(snap.k_ring, snap.v_ring)
                ring_restore_ms = (time.perf_counter() - t_r0) * 1000.0
                if self.engine.hybrid_kv:
                    t_h0 = time.perf_counter()
                    self.engine.restore_global_hot_from_ram(restored_S)
                    hot_kv_restore_ms = (time.perf_counter() - t_h0) * 1000.0
                self.kv_tokens = self.kv_tokens[:restored_S]
                self.last_logits = None
                reprocessed = 0
                suffix = new_tokens[restored_S:]
                t_p0 = time.perf_counter()
                logits = self._process_token_span(suffix, start_pos=restored_S, compute_last_logits=True)
                prefill_ms = (time.perf_counter() - t_p0) * 1000.0
            elif snap is not None and snap.abs_pos > 0:
                case = "C_ROLLBACK_REPROCESS"
                restored_S = snap.abs_pos
                t_r0 = time.perf_counter()
                self.engine.write_ring_kv(snap.k_ring, snap.v_ring)
                ring_restore_ms = (time.perf_counter() - t_r0) * 1000.0
                if self.engine.hybrid_kv:
                    t_h0 = time.perf_counter()
                    self.engine.restore_global_hot_from_ram(restored_S)
                    hot_kv_restore_ms = (time.perf_counter() - t_h0) * 1000.0
                self.kv_tokens = self.kv_tokens[:restored_S]
                self.last_logits = None
                reprocessed = target_p - restored_S
                span_all = new_tokens[restored_S:]
                t_p0 = time.perf_counter()
                logits = self._process_token_span(span_all, start_pos=restored_S, compute_last_logits=True)
                prefill_ms = (time.perf_counter() - t_p0) * 1000.0
            else:
                case = "D_COLD_PREFILL"
                restored_S = None
                reprocessed = 0
                self.engine.clear_kv()
                self.kv_tokens = []
                self.last_logits = None
                t_p0 = time.perf_counter()
                logits = self._process_token_span(new_tokens, start_pos=0, compute_last_logits=True)
                prefill_ms = (time.perf_counter() - t_p0) * 1000.0

        if save_boundary_snapshot:
            self.save_ring_snapshot()

        reused = restored_S if restored_S is not None else (prev_L if case == "A_DIRECT" else 0)
        new_suffix = len(new_tokens) - target_p
        total_proc = len(new_tokens) - reused
        ttft_ms = (time.perf_counter() - t0) * 1000.0

        rep = ContinuityReport(
            case=case, prev_L=prev_L, prompt_len=len(new_tokens), common_prefix_P=P,
            restored_snapshot_S=restored_S, reused_prefix_tokens=reused,
            reprocessed_tokens_S_to_P=reprocessed, new_suffix_tokens=new_suffix,
            total_tokens_processed=total_proc, ring_restore_ms=round(ring_restore_ms, 3),
            prefill_ms=round(prefill_ms, 3), ttft_ms=round(ttft_ms, 3),
            hot_kv_restore_ms=round(hot_kv_restore_ms, 3),
        )
        return logits, rep

    def _check_stop_strings(self, text: str, stop_strings: list[str]) -> str | None:
        for s in stop_strings:
            if s and s in text:
                return s
        return None

    def generate_stream(
        self,
        prompt_tokens: list[int],
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop_strings: list[str] | None = None,
        cancel_event: threading.Event | None = None,
        seed: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        stop_list = [s for s in (stop_strings or []) if s]
        rng = np.random.default_rng(seed)

        with self.lock:
            logits, cont = self.prepare_prompt(prompt_tokens, save_boundary_snapshot=True)
            yield {"type": "meta", "continuity": cont, "prompt_tokens": len(prompt_tokens)}

            parser = HarmonyStreamParser(tokenizer=self.tokenizer)
            generated_ids: list[int] = []
            use_ngram = (temperature <= 0.0) and self.enable_ngram_spec
            ngram_idx = _FastNgramIndex(self.kv_tokens, min_n=5, max_n=6) if use_ngram else None

            next_tok = sample_token(logits, temperature=temperature, top_p=top_p, rng=rng)
            t_gen0 = time.perf_counter()
            finish_reason: str | None = None
            spec_verify_calls = 0
            spec_accepted_total = 0

            def emit_and_check(tid: int):
                nonlocal finish_reason
                generated_ids.append(tid)
                d = parser.feed_token(tid)
                if d.finish_reason is not None:
                    finish_reason = d.finish_reason
                    return d, True
                if stop_list:
                    hit = self._check_stop_strings(parser.final_text, stop_list)
                    if hit is not None:
                        finish_reason = "stop"
                        d.finish_reason = "stop"
                        return d, True
                if len(generated_ids) >= max_tokens:
                    d2 = parser.flush_on_length()
                    finish_reason = "length"
                    if d.content or d.reasoning_content:
                        d.finish_reason = "length"
                        return d, True
                    return d2, True
                return d, False

            delta0, should_stop = emit_and_check(next_tok)
            yield {"type": "delta", "delta": delta0, "token_id": next_tok}

            if should_stop:
                if next_tok == TOK_CALL and len(self.kv_tokens) < self.kv_cap_global:
                    self._process_token_span([next_tok], start_pos=len(self.kv_tokens), compute_last_logits=False)
                self.save_ring_snapshot()
                dt_gen = max(1e-6, time.perf_counter() - t_gen0)
                yield {
                    "type": "done",
                    "finish_reason": finish_reason or "stop",
                    "completion_tokens": len(generated_ids),
                    "generated_ids": generated_ids,
                    "content": parser.final_text,
                    "reasoning_content": parser.analysis_text,
                    "tool_calls": parser.tool_calls,
                    "gen_seconds": round(dt_gen, 4),
                    "gen_tps": round(len(generated_ids) / dt_gen, 3),
                    "spec_verify_calls": spec_verify_calls,
                    "spec_accepted_total": spec_accepted_total,
                    "kv_L": len(self.kv_tokens),
                }
                return

            cur_tok = next_tok
            while len(generated_ids) < max_tokens:
                if cancel_event is not None and cancel_event.is_set():
                    finish_reason = "cancelled"
                    break
                cur_pos = len(self.kv_tokens)
                if cur_pos >= self.kv_cap_global - 1:
                    finish_reason = "length"
                    break

                cands: list[int] = []
                rem = max_tokens - len(generated_ids)
                cur_K = min(self.ngram_k, rem - 1) if rem > 1 else 0
                if use_ngram and ngram_idx is not None and cur_K >= 2 and cur_pos + cur_K + 1 < self.kv_cap_global:
                    tail = (self.kv_tokens[-6:] + [cur_tok])[-6:]
                    ng = ngram_idx.draft(tail, cur_K)
                    for c in ng:
                        if c in HARMONY_STOP_TOKEN_IDS:
                            break
                        cands.append(c)

                if cands:
                    m_k = len(cands)
                    self.engine.ring_snapshot(cur_pos, m_k + 1)
                    verify_seq = [cur_tok] + cands
                    v_logits, _, _, _ = self.engine.verify_tokens(verify_seq, start_pos=cur_pos)
                    spec_verify_calls += 1

                    n_draft_acc = 0
                    corr_tok = -1
                    for j in range(m_k):
                        t_star = int(np.argmax(v_logits[j]))
                        if cands[j] == t_star:
                            n_draft_acc += 1
                        else:
                            corr_tok = t_star
                            break
                    if corr_tok < 0:
                        corr_tok = int(np.argmax(v_logits[m_k]))

                    n_committed = 1 + n_draft_acc
                    self.engine.ring_rollback(cur_pos, first_rejected_i=n_committed, K=m_k + 1)
                    self.engine.commit_verified_cache_accesses(n_committed)
                    committed_slice = verify_seq[:n_committed]
                    self.kv_tokens.extend(committed_slice)
                    if ngram_idx is not None:
                        ngram_idx.extend(committed_slice)
                    self.last_logits = v_logits[n_draft_acc].copy()
                    spec_accepted_total += n_committed

                    emitted = cands[:n_draft_acc] + [corr_tok]
                    stopped = False
                    for ei, tid in enumerate(emitted):
                        d, stp = emit_and_check(tid)
                        yield {"type": "delta", "delta": d, "token_id": tid}
                        if stp:
                            if ei == len(emitted) - 1 and tid == TOK_CALL and len(self.kv_tokens) < self.kv_cap_global:
                                self._process_token_span([tid], start_pos=len(self.kv_tokens), compute_last_logits=False)
                            stopped = True
                            break
                    if stopped:
                        break
                    cur_tok = corr_tok
                else:
                    lg, _ = self.engine.forward_token(int(cur_tok), cur_pos, return_top4=False)
                    self.kv_tokens.append(int(cur_tok))
                    if ngram_idx is not None:
                        ngram_idx.extend([int(cur_tok)])
                    self.last_logits = lg
                    nxt = sample_token(lg, temperature=temperature, top_p=top_p, rng=rng)
                    d, stp = emit_and_check(nxt)
                    yield {"type": "delta", "delta": d, "token_id": nxt}
                    if stp:
                        if nxt == TOK_CALL and len(self.kv_tokens) < self.kv_cap_global:
                            self._process_token_span([nxt], start_pos=len(self.kv_tokens), compute_last_logits=False)
                        break
                    cur_tok = nxt

            self.save_ring_snapshot()
            dt_gen = max(1e-6, time.perf_counter() - t_gen0)
            yield {
                "type": "done",
                "finish_reason": finish_reason or "stop",
                "completion_tokens": len(generated_ids),
                "generated_ids": generated_ids,
                "content": parser.final_text,
                "reasoning_content": parser.analysis_text,
                "tool_calls": parser.tool_calls,
                "gen_seconds": round(dt_gen, 4),
                "gen_tps": round(len(generated_ids) / dt_gen, 3),
                "spec_verify_calls": spec_verify_calls,
                "spec_accepted_total": spec_accepted_total,
                "kv_L": len(self.kv_tokens),
            }

    def close(self) -> None:
        self.engine.close()
