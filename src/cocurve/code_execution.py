"""Sandboxed pass@1 execution scoring for HumanEval / MBPP generations.

The standard-eval generation stage only saves completions
(`generation_saved_no_execution`). This module turns those saved completions
into executable programs, runs their unit tests in an isolated subprocess with
wall-clock and memory limits, and reports pass@1. It is intentionally
self-contained so the same scorer applies identically to the full model, our
pruned model, and every baseline — keeping the comparison fair.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import re
from pathlib import Path
from typing import Dict, List, Sequence

# Sequences that mark the end of the intended completion. Generations often keep
# rambling (new defs, examples, prose); HumanEval-style harnesses truncate at the
# first of these so only the target function body is executed.
_HUMANEVAL_STOP = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\n@", "\nassert "]


def truncate_completion(completion: str, stops: Sequence[str] = _HUMANEVAL_STOP) -> str:
    cut = len(completion)
    for stop in stops:
        idx = completion.find(stop)
        if idx != -1:
            cut = min(cut, idx)
    return completion[:cut]


def _strip_code_fences(text: str) -> str:
    """Pull code out of markdown fences when the model wraps its answer."""
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL)
    if fence:
        return fence.group(1)
    return text


def _fix_body_indent(body: str, target: int = 4) -> str:
    """HumanEval prompts end inside a function (4-space body). Some models
    (notably llama-2-13b) emit the first body line under-indented (e.g. 3 spaces),
    which raises IndentationError and fails EVERY task — a harness artifact, not a
    real coding failure. Pad the first non-empty line up to the function-body
    indent; correctly-indented completions (>=4 spaces) are left untouched."""
    lines = body.split("\n")
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        indent = len(ln) - len(ln.lstrip(" "))
        if indent < target:
            lines[i] = " " * target + ln.lstrip(" ")
        break
    return "\n".join(lines)


def build_humaneval_program(prompt: str, completion: str, test: str, entry_point: str) -> str:
    body = _fix_body_indent(truncate_completion(completion))
    return f"{prompt}{body}\n\n{test}\n\ncheck({entry_point})\n"


def build_mbpp_program(completion: str, tests: Sequence[str]) -> str:
    # 3-shot MBPP generations end the answer at the [DONE] marker (and may then
    # hallucinate the next few-shot block); cut there first.
    code = completion
    for marker in ("[DONE]", "\n[BEGIN]", "\nYou are an expert"):
        idx = code.find(marker)
        if idx != -1:
            code = code[:idx]
    code = _strip_code_fences(code)
    code = truncate_completion(code, ["\n\n\n", "\n# Test", "\nprint("])
    test_block = "\n".join(str(t) for t in tests)
    return f"{code}\n\n{test_block}\n"


def _worker(program: str, queue: "mp.Queue") -> None:  # pragma: no cover - runs in subprocess
    import resource

    # Cap address space (~4 GB) and CPU seconds so a runaway generation cannot
    # take down the host. Best-effort: not all limits apply on every platform.
    try:
        resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 * 1024 * 1024, resource.RLIM_INFINITY))
        resource.setrlimit(resource.RLIMIT_CPU, (10, 12))
    except Exception:
        pass
    import builtins
    import io
    import contextlib

    sandbox_globals: Dict[str, object] = {"__name__": "__candidate__", "__builtins__": builtins}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(compile(program, "<candidate>", "exec"), sandbox_globals)
        queue.put(("passed", ""))
    except Exception as exc:  # noqa: BLE001 - any failure means the candidate failed
        queue.put(("failed", f"{type(exc).__name__}: {exc}"))


def run_program(program: str, timeout: float = 10.0) -> Dict[str, object]:
    """Execute one candidate program in an isolated process with a timeout."""
    ctx = mp.get_context("spawn")
    queue: "mp.Queue" = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(program, queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {"passed": False, "status": "timeout", "detail": f"timeout>{timeout}s"}
    if queue.empty():
        return {"passed": False, "status": "crashed", "detail": f"exit={proc.exitcode}"}
    status, detail = queue.get()
    return {"passed": status == "passed", "status": status, "detail": detail}


def score_humaneval_generations(generations: List[Dict[str, object]], timeout: float = 10.0) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    passed = 0
    for g in generations:
        program = build_humaneval_program(
            prompt=str(g["prompt"]),
            completion=str(g["prediction"]),
            test=str(g["test"]),
            entry_point=str(g["entry_point"]),
        )
        result = run_program(program, timeout=timeout)
        passed += int(bool(result["passed"]))
        rows.append({"task_id": g.get("task_id"), **result})
    total = len(generations)
    return {"task": "humaneval", "metric": "pass@1", "num_samples": total,
            "pass@1": passed / max(1, total), "num_passed": passed, "rows": rows}


def score_mbpp_generations(generations: List[Dict[str, object]], timeout: float = 10.0) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    passed = 0
    for g in generations:
        tests = g.get("tests") or g.get("test_list") or []
        program = build_mbpp_program(completion=str(g["prediction"]), tests=tests)
        result = run_program(program, timeout=timeout)
        passed += int(bool(result["passed"]))
        rows.append({"task_id": g.get("task_id"), **result})
    total = len(generations)
    return {"task": "mbpp", "metric": "pass@1", "num_samples": total,
            "pass@1": passed / max(1, total), "num_passed": passed, "rows": rows}


def score_generation_file(path: Path, timeout: float = 10.0) -> Dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    task = payload.get("task")
    generations = payload.get("generations", [])
    if task == "humaneval":
        return score_humaneval_generations(generations, timeout=timeout)
    if task == "mbpp":
        return score_mbpp_generations(generations, timeout=timeout)
    raise ValueError(f"Unsupported coding task in {path}: {task}")
