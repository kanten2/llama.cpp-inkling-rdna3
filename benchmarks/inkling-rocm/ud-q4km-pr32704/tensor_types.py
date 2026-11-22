"""Inspect routed expert weight types with the gguf-py reader from this checkout."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "gguf-py"))
from gguf import GGUFReader


def inspect(path):
    reader = GGUFReader(str(path))
    groups = defaultdict(list)
    for tensor in reader.tensors:
        match = re.fullmatch(r"blk\.(\d+)\.(ffn_(?:gate_up|gate|up|down)_exps)\.weight", tensor.name)
        if match:
            key = (match[2], tensor.tensor_type.name, tuple(map(int, tensor.shape)))
            groups[key].append(int(match[1]))
    if not groups:
        raise ValueError(f"No routed expert weights in {path.name}")
    return {"filename": path.name, "weights": [
        {"role": role, "type": kind, "shape": shape, "layers": sorted(groups[role, kind, shape])}
        for role, kind, shape in sorted(groups)]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps([inspect(model) for model in args.models], indent=2) + "\n", encoding="utf8")
