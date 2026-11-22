"""Summarize GPU timestamp logs from the single-sequence IQ4_NL decode probe."""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


def read_run(root, name):
    text = (root / f"model-{name}.server.log").read_text(encoding="utf8", errors="replace")
    epochs = {int(e): tuple(map(int, (stripe, stride, khz))) for e, stripe, stride, khz in
              re.findall(r"IQ4_EPOCH (\d+) stripe=(\d+) stride=(\d+) khz=(\d+)", text)}
    sites = defaultdict(dict)
    for line in re.findall(r"IQ4_SITE ([^\r\n]+)", text):
        epoch, site, weight, k, n, experts, active, routed, fused = line.split()
        sites[int(epoch)][int(site)] = (weight, int(k), int(n), int(experts), int(active), int(routed), int(fused))
    rows = []
    captured_sites = None
    visited = defaultdict(set)
    for line in re.findall(r"IQ4_TICKS ([^\r\n]+)", text):
        epoch, replay, site, empty, quant, matvec = map(int, line.split())
        stripe, stride, khz = epochs[epoch]
        assert site % stride == stripe and site not in visited[replay]
        visited[replay].add(site)
        if captured_sites is None:
            captured_sites = sites[epoch]
        assert captured_sites == sites[epoch]
        rows.append((replay, site, empty*1000/khz, quant*1000/khz, matvec*1000/khz))
    assert rows and len(visited) >= 20
    assert all(ids == set(range(stripe, len(captured_sites), stride)) for ids in visited.values())
    return captured_sites, rows, {"sampled_replays": sorted(visited), "epochs_with_samples": sorted(
        {int(v) for v in re.findall(r"IQ4_TICKS (\d+)", text)}), "clock_khz": khz,
        "median_empty_us": median(r[2] for r in rows), "logged_site_samples": len(rows)}


def summarize(root, names):
    # The model and sampling schedule are deterministic across stripes.
    reference_sites = None
    reference_replays = None
    samples = defaultdict(dict)
    runs = {}
    for name in names:
        sites, rows, meta = read_run(root, name)
        if reference_sites is None:
            reference_sites = sites
            reference_replays = meta["sampled_replays"]
        assert sites == reference_sites and meta["sampled_replays"] == reference_replays
        runs[name] = meta
        gap = meta["median_empty_us"]
        for replay, site, _, quant, matvec in rows:
            assert site not in samples[replay]
            samples[replay][site] = (quant, matvec, quant-gap, matvec-gap)

    assert all(set(s) == set(reference_sites) for s in samples.values())
    roles = defaultdict(list)
    for site, (weight, k, n, experts, active, routed, fused) in reference_sites.items():
        role = re.sub(r"^blk\.\d+\.", "", weight)
        roles[(role, k, n, experts, active, routed, fused)].append(site)
    totals = []
    for (role, k, n, experts, active, routed, fused), ids in roles.items():
        per_replay = [[sum(s[site][j] for site in ids) for j in range(4)] for s in samples.values()]
        raw_q, raw_m, net_q, net_m = map(mean, zip(*per_replay))
        m = sorted(row[3] for row in per_replay)
        totals.append({"role": role, "k": k, "n": n, "stored_experts": experts, "active_experts": active,
                       "routed": bool(routed), "fused": bool(fused), "dispatches_per_token": len(ids),
                       "distinct_weights": len({reference_sites[i][0] for i in ids}),
                       "quant_us_raw_per_token": raw_q, "mmvq_us_raw_per_token": raw_m,
                       "quant_us_per_token": net_q, "mmvq_us_per_token": net_m,
                       "mmvq_us_per_dispatch": net_m/len(ids),
                       "mmvq_us_per_token_p10_p90": [m[len(m)//10], m[(len(m)*9)//10]],
                       "visits_by_weight": dict(Counter(reference_sites[i][0] for i in ids))})
    total_m = sum(x["mmvq_us_per_token"] for x in totals)
    for x in totals:
        x["share_of_iq4_mmvq_pct"] = 100*x["mmvq_us_per_token"]/total_m
    return {"runs": runs, "aligned_replays": len(samples), "sites_per_token": len(reference_sites),
            "total_mmvq_us_per_token": total_m, "shapes": sorted(totals, key=lambda x:-x["mmvq_us_per_token"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    root = args.folder
    report = {"striped": summarize(root, [f"stripe{i}" for i in range(4)]), "dense": summarize(root, ["dense"])}
    ref = json.loads((root/"official-live.json").read_text(encoding="utf8"))["choices"][0]["message"]
    report["requests"] = {}
    for name in ["baseline-a", "stripe0", "stripe1", "stripe2", "stripe3", "dense", "baseline-b"]:
        result = json.loads((root/f"model-{name}.trial-1.json").read_text(encoding="utf8"))
        summary = json.loads((root/f"model-{name}.summary.json").read_text(encoding="utf8"))
        t = result["timings"]
        assert (t["cache_n"], t["prompt_n"], t["predicted_n"]) == (0, 103, 768)
        assert result["choices"][0]["message"] == ref
        report["requests"][name] = {"tok_s": t["predicted_per_second"], "ms_per_token": t["predicted_per_token_ms"],
                                    "peak_dedicated_gib": summary["trials"][0]["vram"]["dedicated"]["max"]/2**30,
                                    "identical_to_reference": True}
    baseline_ms = mean(report["requests"][n]["ms_per_token"] for n in ("baseline-a", "baseline-b"))
    report["baseline_mean_ms_per_token"] = baseline_ms
    report["iq4_mmvq_share_of_baseline_decode_pct"] = 100*report["striped"]["total_mmvq_us_per_token"]/(1000*baseline_ms)
    path = root/"shape-profile.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf8")
    for variant in ("striped","dense"):
        print(variant, report[variant]["aligned_replays"], [(s["role"], round(s["mmvq_us_per_token"], 1),
              round(s["quant_us_per_token"], 1)) for s in report[variant]["shapes"]])


if __name__ == "__main__":
    main()
