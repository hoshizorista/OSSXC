"""Download (optional) and pack official openai/gpt-oss-20b safetensors for OSSXC.

Creates strictly untouched 4-bit native MXFP4 streaming binaries + resident attention/head weights:
  - <weights_dir>/resident_gpu.pt                  (~3.63 GB: embed_tokens, lm_head, norms, attn, router, biases)
  - <weights_dir>/mxfp4_experts/layer_00..23.bin   (~10.15 GB: 24 x 423,014,400 B native MXFP4 blocks + E8M0 scales)
  - <weights_dir>/mxfp4_experts/index.json         (exact byte-offset metadata)

Zero quantization is applied to the target model; MXFP4 nibbles and E8M0 scales are copied verbatim.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from ossxc.config import (
    EXPERT_BYTES,
    HF_MODEL_DIR,
    HF_MODEL_REPO,
    LAYER_BYTES,
    NUM_EXPERTS,
    NUM_LAYERS,
    WEIGHTS_DIR,
)


class SafetensorsReader:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        idx_path = model_dir / "model.safetensors.index.json"
        if not idx_path.is_file():
            raise FileNotFoundError(
                f"Missing {idx_path}. Run with --download to fetch {HF_MODEL_REPO} from Hugging Face."
            )
        with open(idx_path, "r", encoding="utf-8") as f:
            self.weight_map: dict[str, str] = json.load(f)["weight_map"]
        self._handles: dict[str, object] = {}

    def get(self, name: str) -> torch.Tensor:
        fname = self.weight_map[name]
        if fname not in self._handles:
            self._handles[fname] = safe_open(str(self.model_dir / fname), framework="pt", device="cpu")
        return self._handles[fname].get_tensor(name)

    def close(self) -> None:
        self._handles.clear()


def _append_aligned16(buf: bytearray, data: bytes) -> tuple[int, int]:
    off = len(buf)
    buf.extend(data)
    rem = len(buf) & 15
    if rem:
        buf.extend(b"\x00" * (16 - rem))
    return off, len(data)


def download_model(hf_dir: Path, repo_id: str) -> None:
    from huggingface_hub import snapshot_download

    hf_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1/3] Downloading {repo_id} into {hf_dir} ...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(hf_dir),
        allow_patterns=[
            "*.safetensors",
            "model.safetensors.index.json",
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.jinja",
        ],
    )
    print("  Download complete.")


def pack_weights(hf_dir: Path, out_dir: Path) -> None:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    mxfp4_dir = out_dir / "mxfp4_experts"
    mxfp4_dir.mkdir(parents=True, exist_ok=True)

    reader = SafetensorsReader(hf_dir)
    try:
        print(f"[2/3] Packing resident attention, router, embeddings & lm_head -> {out_dir / 'resident_gpu.pt'} ...")
        resident: dict = {
            "embed_tokens": reader.get("model.embed_tokens.weight").to(torch.float16).contiguous(),
            "lm_head": reader.get("lm_head.weight").to(torch.float16).contiguous(),
            "final_norm": reader.get("model.norm.weight").to(torch.float32).contiguous(),
            "layers": [],
        }

        layers_meta = []
        print(f"[3/3] Packing 24 native MXFP4 expert layers -> {mxfp4_dir} ...")
        for l in range(NUM_LAYERS):
            p = f"model.layers.{l}"
            gu_bias = reader.get(f"{p}.mlp.experts.gate_up_proj_bias").to(torch.float16).contiguous()
            layer_res = {
                "in_norm": reader.get(f"{p}.input_layernorm.weight").to(torch.float32).contiguous(),
                "post_norm": reader.get(f"{p}.post_attention_layernorm.weight").to(torch.float32).contiguous(),
                "q_w": reader.get(f"{p}.self_attn.q_proj.weight").to(torch.float16).contiguous(),
                "q_b": reader.get(f"{p}.self_attn.q_proj.bias").to(torch.float16).contiguous(),
                "k_w": reader.get(f"{p}.self_attn.k_proj.weight").to(torch.float16).contiguous(),
                "k_b": reader.get(f"{p}.self_attn.k_proj.bias").to(torch.float16).contiguous(),
                "v_w": reader.get(f"{p}.self_attn.v_proj.weight").to(torch.float16).contiguous(),
                "v_b": reader.get(f"{p}.self_attn.v_proj.bias").to(torch.float16).contiguous(),
                "o_w": reader.get(f"{p}.self_attn.o_proj.weight").to(torch.float16).contiguous(),
                "o_b": reader.get(f"{p}.self_attn.o_proj.bias").to(torch.float16).contiguous(),
                "sinks": reader.get(f"{p}.self_attn.sinks").to(torch.float32).contiguous(),
                "router_w": reader.get(f"{p}.mlp.router.weight").to(torch.float16).contiguous(),
                "router_b": reader.get(f"{p}.mlp.router.bias").to(torch.float16).contiguous(),
                "gate_bias": gu_bias[:, 0::2].contiguous(),
                "up_bias": gu_bias[:, 1::2].contiguous(),
                "down_bias": reader.get(f"{p}.mlp.experts.down_proj_bias").to(torch.float16).contiguous(),
            }
            resident["layers"].append(layer_res)

            # Pack 32 untouched MXFP4 experts for layer l
            ep = f"{p}.mlp.experts"
            gu_b = reader.get(f"{ep}.gate_up_proj_blocks").to(torch.uint8).contiguous().numpy()
            gu_s = reader.get(f"{ep}.gate_up_proj_scales").to(torch.uint8).contiguous().numpy()
            dn_b = reader.get(f"{ep}.down_proj_blocks").to(torch.uint8).contiguous().numpy()
            dn_s = reader.get(f"{ep}.down_proj_scales").to(torch.uint8).contiguous().numpy()

            buf = bytearray()
            experts_meta = []
            for e in range(NUM_EXPERTS):
                e0 = len(buf)
                off_gub, _ = _append_aligned16(buf, gu_b[e].tobytes())
                off_gus, _ = _append_aligned16(buf, gu_s[e].tobytes())
                off_dnb, _ = _append_aligned16(buf, dn_b[e].tobytes())
                off_dns, _ = _append_aligned16(buf, dn_s[e].tobytes())
                esize = len(buf) - e0
                if esize != EXPERT_BYTES:
                    raise RuntimeError(f"Layer {l} expert {e}: expected {EXPERT_BYTES} B, got {esize} B")
                experts_meta.append({
                    "offset": e0,
                    "size": esize,
                    "gu_blocks_off": off_gub - e0,
                    "gu_scales_off": off_gus - e0,
                    "dn_blocks_off": off_dnb - e0,
                    "dn_scales_off": off_dns - e0,
                })

            if len(buf) != LAYER_BYTES:
                raise RuntimeError(f"Layer {l}: expected {LAYER_BYTES} B, got {len(buf)} B")

            bin_path = mxfp4_dir / f"layer_{l:02d}.bin"
            with open(bin_path, "wb") as f:
                f.write(buf)
            layers_meta.append({
                "layer_idx": l,
                "file_bytes": len(buf),
                "experts": experts_meta,
            })
            del gu_b, gu_s, dn_b, dn_s, buf, gu_bias
            gc.collect()
            print(f"  layer_{l:02d}.bin packed ({LAYER_BYTES / 1e6:.2f} MB)")

        res_path = out_dir / "resident_gpu.pt"
        torch.save(resident, str(res_path))
        total_mxfp4 = sum(x["file_bytes"] for x in layers_meta)
        idx_obj = {
            "variant": "ossxc_mxfp4_native",
            "expert_bytes": EXPERT_BYTES,
            "layer_bytes": LAYER_BYTES,
            "total_bytes": total_mxfp4,
            "layers": layers_meta,
        }
        with open(mxfp4_dir / "index.json", "w", encoding="utf-8") as f:
            json.dump(idx_obj, f, indent=2)

        dt = time.perf_counter() - t0
        res_gb = os.path.getsize(res_path) / 1e9
        print(
            f"\nDone in {dt:.1f}s!\n"
            f"  resident_gpu.pt : {res_gb:.2f} GB\n"
            f"  mxfp4_experts/  : {total_mxfp4 / 1e9:.2f} GB (24 x {LAYER_BYTES / 1e6:.2f} MB, 100% untouched MXFP4)"
        )
    finally:
        reader.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare openai/gpt-oss-20b native MXFP4 weights for OSSXC.")
    ap.add_argument("--hf-dir", default=HF_MODEL_DIR, help=f"Path to Hugging Face checkpoint (default: {HF_MODEL_DIR})")
    ap.add_argument("--out-dir", default=WEIGHTS_DIR, help=f"Output weights directory (default: {WEIGHTS_DIR})")
    ap.add_argument("--repo-id", default=HF_MODEL_REPO, help=f"HF Hub repo ID (default: {HF_MODEL_REPO})")
    ap.add_argument("--download", action="store_true", help="Download safetensors + tokenizer from Hugging Face first")
    args = ap.parse_args()

    hf_dir = Path(args.hf_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if args.download:
        download_model(hf_dir, args.repo_id)
    pack_weights(hf_dir, out_dir)


if __name__ == "__main__":
    main()
