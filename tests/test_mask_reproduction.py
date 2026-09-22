import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).parents[1] / "scripts" / "reproduce_paper_mask.py"
_SPEC = importlib.util.spec_from_file_location("reproduce_paper_mask", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


def test_reproduces_representative_llm_and_vlm_masks():
    assert _MODULE.reproduce("llama-3.1-8b-instruct", 30)["match"]
    assert _MODULE.reproduce("qwen2.5-vl-7b", 40)["match"]
    # This operating point guards the boundary-representative provenance path.
    assert _MODULE.reproduce("mistral-small-24b", 50)["match"]
