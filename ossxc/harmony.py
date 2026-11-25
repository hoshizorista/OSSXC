"""Official Harmony prompt renderer and streaming parser for openai/gpt-oss-20b.
Hardened for agentic OpenAI-compatible clients such as OpenCode (@ai-sdk/openai-compatible):
  - Sanitizes complex Zod/JSON-Schema tool definitions (anyOf -> oneOf, non-string enum defaults, null descriptions)
  - Interleaves multi-tool-call assistant turns + multiple tool responses into exact 1:1 Harmony pairs
  - Guards against orphaned tool messages and channel tag leakage
  - Parses analysis / commentary / final channels and function tool calls cleanly during SSE streaming
"""
from __future__ import annotations

import copy
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from transformers import AutoTokenizer

TOK_STARTOFTEXT = 199998
TOK_ENDOFTEXT = 199999
TOK_RETURN = 200002
TOK_CONSTRAIN = 200003
TOK_CHANNEL = 200005
TOK_START = 200006
TOK_END = 200007
TOK_MESSAGE = 200008
TOK_CALL = 200012

HARMONY_STOP_TOKEN_IDS = frozenset({TOK_RETURN, TOK_CALL, TOK_ENDOFTEXT})

_RE_CHANNEL_TAG = re.compile(r"<\|channel\|>(?:analysis|commentary|final)<\|message\|>")


def load_official_tokenizer(model_dir: str):
    return AutoTokenizer.from_pretrained(model_dir, local_files_only=True)


def _normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in ("text", "input_text", "output_text"):
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    parts.append(_normalize_content(item["content"]))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def _sanitize_param_schema(spec: Any) -> Any:
    """Recursively sanitize JSON Schema objects for openai/gpt-oss-20b chat_template.jinja."""
    if not isinstance(spec, dict):
        return spec
    out = copy.deepcopy(spec)
    # Normalize JSON Schema `anyOf` -> `oneOf` (supported by Harmony Jinja template)
    if "anyOf" in out and "oneOf" not in out:
        out["oneOf"] = out.pop("anyOf")
    # Ensure enum values are strings and default is a string when enum/oneOf is present
    # (official chat_template.jinja concatenates `", // default: " + param_spec.default`)
    if "enum" in out and isinstance(out["enum"], list):
        out["enum"] = [str(v) for v in out["enum"]]
        if "default" in out and not isinstance(out["default"], str):
            out["default"] = json.dumps(out["default"])
    if "oneOf" in out and isinstance(out["oneOf"], list):
        out["oneOf"] = [_sanitize_param_schema(v) for v in out["oneOf"]]
        if "default" in out and not isinstance(out["default"], str):
            out["default"] = json.dumps(out["default"])
    if "items" in out and isinstance(out["items"], dict):
        out["items"] = _sanitize_param_schema(out["items"])
    if "properties" in out and isinstance(out["properties"], dict):
        out["properties"] = {k: _sanitize_param_schema(v) for k, v in out["properties"].items()}
    if "description" in out and out["description"] is None:
        out["description"] = ""
    return out


def _normalize_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    norm = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        t_copy = copy.deepcopy(t)
        if "function" not in t_copy and "name" in t_copy:
            t_copy = {"type": "function", "function": t_copy}
        fn = t_copy.get("function")
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        if not isinstance(fn.get("description"), str):
            fn["description"] = "" if fn.get("description") is None else str(fn["description"])
        params = fn.get("parameters")
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        else:
            params = _sanitize_param_schema(params)
            if "type" not in params:
                params["type"] = "object"
            if "properties" not in params or not isinstance(params["properties"], dict):
                params["properties"] = {}
        fn["parameters"] = params
        t_copy["type"] = "function"
        t_copy["function"] = fn
        norm.append(t_copy)
    return norm or None


def _parse_tool_args(raw_args: Any) -> Any:
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except Exception:
            return {"raw": raw_args}
    return raw_args if raw_args is not None else {}


def _parse_tool_result(raw_content: Any) -> Any:
    txt = _normalize_content(raw_content)
    try:
        return json.loads(txt)
    except Exception:
        return txt


def normalize_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize OpenAI/OpenCode chat messages for official openai/gpt-oss-20b chat_template.jinja.
    Safely interleaves multi-tool-call assistant turns into 1:1 (assistant, tool) pairs and
    handles orphaned tool outputs without throwing Jinja template exceptions.
    """
    system_parts: list[str] = []
    non_sys: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user")
        if role in ("system", "developer"):
            txt = _normalize_content(m.get("content"))
            if txt:
                system_parts.append(txt)
        else:
            non_sys.append(m)

    rest: list[dict[str, Any]] = []
    i = 0
    n = len(non_sys)
    last_had_tool_call = False

    while i < n:
        m = non_sys[i]
        role = m.get("role", "user")

        if role == "assistant":
            content = _RE_CHANNEL_TAG.sub("", _normalize_content(m.get("content")))
            thinking = _RE_CHANNEL_TAG.sub("", _normalize_content(m.get("thinking") or m.get("reasoning_content") or ""))
            tool_calls = m.get("tool_calls") or []

            if tool_calls:
                norm_tcs = []
                for tc in tool_calls:
                    fn = tc.get("function", tc)
                    norm_tcs.append({
                        "id": tc.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                        "type": "function",
                        "function": {
                            "name": fn.get("name", "unknown"),
                            "arguments": _parse_tool_args(fn.get("arguments", {})),
                        },
                    })

                if len(norm_tcs) == 1:
                    msg: dict[str, Any] = {"role": "assistant", "tool_calls": norm_tcs}
                    if thinking:
                        msg["thinking"] = thinking
                    elif content:
                        msg["content"] = content
                    rest.append(msg)
                    last_had_tool_call = True
                    i += 1
                else:
                    # Collect contiguous following `role == "tool"` messages and interleave 1:1
                    j = i + 1
                    tool_msgs: list[dict[str, Any]] = []
                    while j < n and non_sys[j].get("role") == "tool":
                        tool_msgs.append(non_sys[j])
                        j += 1
                    by_id = {tm.get("tool_call_id"): tm for tm in tool_msgs if tm.get("tool_call_id")}
                    used_ids: set[int] = set()
                    for idx_tc, tc in enumerate(norm_tcs):
                        amsg: dict[str, Any] = {"role": "assistant", "tool_calls": [tc]}
                        if idx_tc == 0:
                            if thinking:
                                amsg["thinking"] = thinking
                            elif content:
                                amsg["content"] = content
                        rest.append(amsg)
                        last_had_tool_call = True
                        matched_tm = by_id.get(tc["id"])
                        if matched_tm is None and idx_tc < len(tool_msgs) and id(tool_msgs[idx_tc]) not in used_ids:
                            matched_tm = tool_msgs[idx_tc]
                        if matched_tm is not None:
                            used_ids.add(id(matched_tm))
                            rest.append({"role": "tool", "content": _parse_tool_result(matched_tm.get("content"))})
                    # Any extra unmatched tool results stay safely as tool messages
                    for tm in tool_msgs:
                        if id(tm) not in used_ids:
                            rest.append({"role": "tool", "content": _parse_tool_result(tm.get("content"))})
                    i = j
                continue

            msg = {"role": "assistant", "content": content}
            if thinking:
                msg["thinking"] = thinking
            rest.append(msg)
            last_had_tool_call = False
            i += 1
        elif role == "tool":
            if last_had_tool_call:
                rest.append({"role": "tool", "content": _parse_tool_result(m.get("content"))})
            else:
                # Guard against orphaned tool messages after compaction/history edits
                name = m.get("name") or m.get("tool_call_id") or "tool"
                rest.append({"role": "user", "content": f"[Tool output from {name}]\n{_normalize_content(m.get('content'))}"})
            i += 1
        else:
            rest.append({"role": "user", "content": _normalize_content(m.get("content"))})
            last_had_tool_call = False
            i += 1

    out: list[dict[str, Any]] = []
    if system_parts:
        out.append({"role": "developer", "content": "\n\n".join(system_parts)})
    out.extend(rest)
    return out


def render_harmony_prompt_ids(
    tokenizer,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    reasoning_effort: str = "medium",
    model_identity: str | None = None,
) -> tuple[list[int], str]:
    norm_msgs = normalize_openai_messages(messages)
    norm_tools = _normalize_tools(tools)

    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "reasoning_effort": reasoning_effort,
    }
    if norm_tools:
        kwargs["tools"] = norm_tools
    if model_identity:
        kwargs["model_identity"] = model_identity

    prompt_str = tokenizer.apply_chat_template(norm_msgs, **kwargs)
    token_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    return token_ids, prompt_str


@dataclass
class HarmonyDelta:
    channel: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_call: dict[str, Any] | None = None
    finish_reason: str | None = None


@dataclass
class HarmonyStreamParser:
    """Stateful parser for openai/gpt-oss-20b Harmony token streams."""
    tokenizer: Any
    state: str = "HEADER"
    header_tokens: list[int] = field(default_factory=list)
    body_tokens: list[int] = field(default_factory=list)
    current_channel: str = "final"
    current_recipient: str | None = None
    emitted_body_len: int = 0

    analysis_text: str = ""
    final_text: str = ""
    commentary_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None

    def _parse_header(self) -> None:
        hdr = self.tokenizer.decode(self.header_tokens, skip_special_tokens=False)
        ch = "final"
        m_ch = re.search(r"<\|channel\|>\s*([a-zA-Z0-9_]+)", hdr)
        if m_ch:
            ch = m_ch.group(1).strip()
        elif "analysis" in hdr:
            ch = "analysis"
        elif "commentary" in hdr:
            ch = "commentary"
        self.current_channel = ch

        rec = None
        m_to = re.search(r"to=([a-zA-Z0-9_.-]+)", hdr)
        if m_to:
            rec = m_to.group(1).strip()
        self.current_recipient = rec

    def _finalize_body(self, end_tok: int) -> HarmonyDelta:
        full_body = self.tokenizer.decode(self.body_tokens, skip_special_tokens=True)
        delta_text = full_body[self.emitted_body_len:]
        out = HarmonyDelta(channel=self.current_channel)

        if self.current_channel == "analysis":
            if delta_text:
                out.reasoning_content = delta_text
                self.analysis_text += delta_text
        elif self.current_channel == "final":
            if delta_text:
                out.content = delta_text
                self.final_text += delta_text
        elif self.current_channel == "commentary":
            self.commentary_text += full_body
            if end_tok == TOK_CALL or (self.current_recipient and self.current_recipient.startswith("functions.")):
                fn_name = self.current_recipient or "unknown"
                if fn_name.startswith("functions."):
                    fn_name = fn_name[len("functions."):]
                args_str = full_body.strip()
                try:
                    parsed = json.loads(args_str)
                    args_str = json.dumps(parsed, ensure_ascii=False)
                except Exception:
                    pass
                tc = {
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {"name": fn_name, "arguments": args_str},
                }
                self.tool_calls.append(tc)
                out.tool_call = tc

        if end_tok == TOK_CALL:
            self.finish_reason = "tool_calls"
            out.finish_reason = "tool_calls"
        elif end_tok in (TOK_RETURN, TOK_ENDOFTEXT):
            self.finish_reason = "stop"
            out.finish_reason = "stop"

        self.state = "HEADER"
        self.header_tokens.clear()
        self.body_tokens.clear()
        self.emitted_body_len = 0
        self.current_recipient = None
        return out

    def feed_token(self, tok_id: int) -> HarmonyDelta:
        if tok_id == TOK_START:
            self.state = "HEADER"
            self.header_tokens.clear()
            self.body_tokens.clear()
            self.emitted_body_len = 0
            return HarmonyDelta()

        if self.state == "HEADER":
            if tok_id == TOK_MESSAGE:
                self._parse_header()
                self.state = "BODY"
                self.body_tokens.clear()
                self.emitted_body_len = 0
                return HarmonyDelta(channel=self.current_channel)
            if tok_id in HARMONY_STOP_TOKEN_IDS:
                self.finish_reason = "tool_calls" if tok_id == TOK_CALL else "stop"
                return HarmonyDelta(finish_reason=self.finish_reason)
            self.header_tokens.append(tok_id)
            return HarmonyDelta()

        if tok_id in (TOK_END, TOK_RETURN, TOK_CALL, TOK_ENDOFTEXT):
            return self._finalize_body(tok_id)

        self.body_tokens.append(tok_id)
        if self.current_channel in ("analysis", "final"):
            full_body = self.tokenizer.decode(self.body_tokens, skip_special_tokens=True)
            if full_body.endswith("\ufffd"):
                return HarmonyDelta(channel=self.current_channel)
            delta_text = full_body[self.emitted_body_len:]
            if delta_text:
                self.emitted_body_len = len(full_body)
                if self.current_channel == "analysis":
                    self.analysis_text += delta_text
                    return HarmonyDelta(channel="analysis", reasoning_content=delta_text)
                self.final_text += delta_text
                return HarmonyDelta(channel="final", content=delta_text)
        return HarmonyDelta(channel=self.current_channel)

    def flush_on_length(self) -> HarmonyDelta:
        self.finish_reason = "length"
        if self.state == "BODY" and self.body_tokens and self.current_channel in ("analysis", "final"):
            full_body = self.tokenizer.decode(self.body_tokens, skip_special_tokens=True)
            delta_text = full_body[self.emitted_body_len:]
            if delta_text:
                self.emitted_body_len = len(full_body)
                if self.current_channel == "analysis":
                    self.analysis_text += delta_text
                    return HarmonyDelta(channel="analysis", reasoning_content=delta_text, finish_reason="length")
                self.final_text += delta_text
                return HarmonyDelta(channel="final", content=delta_text, finish_reason="length")
        return HarmonyDelta(finish_reason="length")
