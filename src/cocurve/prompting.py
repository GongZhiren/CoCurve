from __future__ import annotations

from typing import Dict, List, Tuple

from .model import ModelBundle


def detect_task_type(record: Dict[str, object]) -> str:
    raw = str(record.get("task_type", "")).lower()
    if raw in {"multiple_choice", "factual", "reasoning", "code"}:
        return raw
    text = " ".join(str(record.get(k, "")) for k in ("question", "prompt", "query")).lower()
    if any(k in text for k in ("a)", "b)", "option", "choices")):
        return "multiple_choice"
    if any(k in text for k in ("python", "code", "function", "implement")):
        return "code"
    if any(k in text for k in ("why", "proof", "derive", "reason")):
        return "reasoning"
    return "factual"


def default_system_prompt(family: str) -> str:
    if family == "qwen":
        return "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    return "You are a helpful assistant."


def build_messages(prompt: str, task_type: str, family: str) -> List[Dict[str, str]]:
    system = default_system_prompt(family)
    task_constraints = {
        "multiple_choice": "Return only the final option letter (for example A/B/C/D).",
        "code": "Return executable code only, no additional explanation.",
        "reasoning": "Provide concise reasoning and end with the final answer.",
        "factual": "Return a concise factual answer.",
    }
    user_text = f"{prompt}\n\nInstruction: {task_constraints.get(task_type, task_constraints['factual'])}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user_text}]


def default_use_chat_template(family: str) -> bool:
    # Base checkpoints are not instruction-tuned; chat templates hurt MC/generation
    # smoke tests. We evaluate gemma/phi as base models, so treat them like llama base.
    return family not in {"llama", "gemma", "phi"}


def render_prompt(
    bundle: ModelBundle,
    prompt: str,
    task_type: str,
    use_chat_template: bool | None = None,
) -> Tuple[str, List[Dict[str, str]]]:
    messages = build_messages(prompt=prompt, task_type=task_type, family=bundle.family)
    tokenizer = bundle.tokenizer
    use_template = default_use_chat_template(bundle.family) if use_chat_template is None else use_chat_template
    if use_template and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return rendered, messages
    return messages[-1]["content"], messages
