"""Grade final answers independently of the inference server. Python 3.9+."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from statistics import mean, median
import unicodedata


HERE = Path(__file__).resolve().parent
FIELDS = {"M": "計算", "L": "論理", "C": "コード", "D": "データ集計", "P": "仕様の読み取り"}
MODES = ("eight", "four")


def read_suite():
    protocol = json.loads((HERE / "protocol.json").read_bytes())
    raw = (HERE / "suite.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != protocol["suite_sha256"]:
        raise ValueError("The question set differs from the frozen answer key")
    suite = {q["id"]: q for q in json.loads(raw)}
    planned = [qid for _, qids, _ in protocol["block_order"] for qid in qids]
    if len(planned) != 25 or set(planned) != set(suite) or len(suite) != 25:
        raise ValueError("Question IDs differ from the registered protocol")
    return protocol, suite


def decode_answer(answer):
    text = unicodedata.normalize("NFKC", answer.strip())
    text = re.sub(r"^```(?:json|text)?\s*|\s*```$", "", text, flags=re.I).strip()
    for line in [text, *reversed(text.splitlines())]:
        line = line.strip().replace("**", "").strip("`").strip()
        line = re.sub(r"^(?:最終回答|回答|答え|Answer)\s*[:=]\s*", "", line, flags=re.I)
        line = line.rstrip("。").strip()
        if re.fullmatch(r"[+-]?\d+", line):
            return int(line)
        if re.fullmatch(r"\[\s*[+-]?\d+(?:\s*,\s*[+-]?\d+)*\s*\]", line):
            return list(map(int, re.findall(r"[+-]?\d+", line)))
    return None


def read_pairs(path, protocol, suite):
    attempts = {}
    blocks = {qid: block for block, ids, _ in protocol["block_order"] for qid in ids}
    tiers = protocol["ctx_tiers"]
    for line in path.read_text(encoding="utf8").splitlines():
        r = json.loads(line)
        qid, mode, ctx = r["id"], r["mode"], r["ctx"]
        key = qid, mode, ctx
        t, u = r["timings"], r["usage"]
        if (key in attempts or mode not in MODES or ctx not in tiers or blocks.get(qid) != r["block"]
                or r["finish"] not in ("stop", "length") or t["cache_n"] != 0
                or t["predicted_n"] != u["completion_tokens"] or t["prompt_n"] != u["prompt_tokens"]
                or t["predicted_ms"] <= 0 or r["wall_s"] <= 0):
            raise ValueError(f"Invalid or duplicate measurement: {key}")
        attempts[key] = r
    rows = []
    truncated = []
    for qid, q in suite.items():
        for ctx in tiers:
            pair = [attempts.pop((qid, mode, ctx)) for mode in MODES]
            if pair[0]["usage"]["prompt_tokens"] != pair[1]["usage"]["prompt_tokens"]:
                raise ValueError(f"Prompt length differs for {qid} at {ctx}")
            cuts = [(qid, ctx, r["mode"]) for r in pair if r["finish"] == "length"]
            if cuts:
                truncated.extend(cuts)
                continue
            row = {"id": qid, "title": q["title"], "expected": q["expected"]}
            for r in pair:
                row[r["mode"]] = dict(r, correct=decode_answer(r["answer"]) == q["expected"])
            rows.append(row)
            break
        else:
            raise ValueError(f"No complete matched pair for {qid}")
    if attempts:
        raise ValueError("Unexpected attempts after matched natural stops")
    rows.sort(key=lambda r: (list(FIELDS).index(r["id"][0]), int(r["id"][1:])))
    return rows, truncated


def aggregate(rows, mode):
    runs = [r[mode] for r in rows]
    seconds = [r["wall_s"] for r in runs]
    tokens = sum(r["usage"]["completion_tokens"] for r in runs)
    return {"n": len(rows), "correct": sum(r["correct"] for r in runs),
            "wall_sum_s": sum(seconds), "wall_mean_s": mean(seconds), "wall_median_s": median(seconds),
            "completion_tokens": tokens,
            "aggregate_decode_tok_s": 1000 * tokens / sum(r["timings"]["predicted_ms"] for r in runs)}


def grade(answers, json_path=None, markdown_path=None):
    protocol, suite = read_suite()
    rows, cuts = read_pairs(answers, protocol, suite)
    same = [r for r in rows if r["eight"]["correct"] and r["four"]["correct"]]
    patterns = Counter((r["eight"]["correct"], r["four"]["correct"]) for r in rows)
    summary = {
        "all": {m: aggregate(rows, m) for m in MODES},
        "both_correct": {m: aggregate(same, m) for m in MODES} if same else {},
        "by_category": {label: {m: sum(r[m]["correct"] for r in rows if r["id"].startswith(k))
                                for m in MODES} for k, label in FIELDS.items()},
        "four_faster_count": sum(r["four"]["wall_s"] < r["eight"]["wall_s"] for r in rows),
        "win_loss": {"both_right": patterns[True, True], "eight_only": patterns[True, False],
                     "four_only": patterns[False, True], "both_wrong": patterns[False, False]},
        "truncations": cuts,
    }
    if json_path:
        json_path.write_bytes((json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf8"))
    if markdown_path:
        lines = ["# Japanese 25-question results", "",
                 "| Mode | Correct | Total seconds | Mean seconds | Median seconds | Generated tokens | Decode tok/s |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for mode, s in summary["all"].items():
            lines.append(f'| {mode} | {s["correct"]}/{s["n"]} | {s["wall_sum_s"]:.2f} | '
                         f'{s["wall_mean_s"]:.2f} | {s["wall_median_s"]:.2f} | {s["completion_tokens"]} | '
                         f'{s["aggregate_decode_tok_s"]:.2f} |')
        lines += ["", "| ID | Problem | Expected | Eight-wave final answer | Seconds | Four-wave final answer | Seconds |",
                  "|---|---|---|---|---:|---|---:|"]
        for r in rows:
            cells = [r["id"], r["title"], json.dumps(r["expected"])]
            for mode in MODES:
                answer = r[mode]["answer"].strip().replace("|", "&#124;").replace("\n", " ")
                cells.extend([("✓ " if r[mode]["correct"] else "✗ ") + answer, f'{r[mode]["wall_s"]:.2f}'])
            lines.append("| " + " | ".join(cells) + " |")
        markdown_path.write_bytes(("\n".join(lines) + "\n").encode("utf8"))
    for m, s in summary["all"].items():
        print(f'{m}: {s["correct"]}/{s["n"]}, {s["wall_sum_s"]:.2f} s, {s["completion_tokens"]} tokens')
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("answers", type=Path)
    parser.add_argument("--json", type=Path, help="Write aggregate results")
    parser.add_argument("--markdown", type=Path, help="Write the per-question comparison")
    args = parser.parse_args()
    grade(args.answers, args.json, args.markdown)
