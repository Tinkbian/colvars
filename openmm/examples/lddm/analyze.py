"""Summarize bidirectional work and speed for completed LDDM legs."""

import csv
import json
from collections import defaultdict

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logsumexp

from protocol import KT, standard_state


def analyze_leg(directories):
    samples = defaultdict(list)
    elapsed, steps = 0., 0
    for i, directory in enumerate(directories):
        result = json.loads((directory/"result.json").read_text())
        if result["index"] != i or result["elapsed_s"] <= 0:
            raise ValueError(f"invalid result {directory}")
        elapsed += result["elapsed_s"]
        steps += result["steps"]
        with (directory/"samples.csv").open() as stream:
            for row in csv.DictReader(stream):
                a, b = int(row["from"]), int(row["to"])
                if a != i or abs(a-b) != 1:
                    raise ValueError(f"invalid state pair {a}, {b}")
                samples[a, b].append(float(row["total_kcal"])/KT)
    edges = []
    for i in range(len(directories)-1):
        fw, bw = (np.asarray(samples[pair]) for pair in ((i, i+1), (i+1, i)))
        if not len(fw) or not len(bw) or not all(np.isfinite(x).all() for x in (fw, bw)):
            raise ValueError(f"missing or nonfinite work on edge {i}")
        ratio = np.log(len(fw)/len(bw))
        def residual(delta):
            return expit(delta-ratio-fw).sum()-expit(ratio-delta-bw).sum()
        lower, upper = min(fw.min(), -bw.max()), max(fw.max(), -bw.min())
        delta = brentq(residual, lower-abs(ratio)-10, upper+abs(ratio)+10)
        forward = -KT*(logsumexp(-fw)-np.log(len(fw)))
        reverse = -KT*(logsumexp(-bw)-np.log(len(bw)))
        edges.append(dict(edge=i, n_forward=len(fw), n_reverse=len(bw), bar=KT*delta,
                          fep=(forward-reverse)/2, hysteresis=forward+reverse))
    return dict(edges=edges, bar=sum(e["bar"] for e in edges), fep=sum(e["fep"] for e in edges),
                steps=steps, elapsed_s=elapsed, ns_day=steps*0.000002*86400/elapsed)


def summarize(root, intervals, modes=("cpu", "gpu", "cupotential")):
    params = json.loads((root/"eq/restraints.json").read_text())
    results = {}
    for mode in modes:
        legs = {}
        for leg, count in zip(("bound", "solvent"), intervals):
            legs[leg] = analyze_leg([root/mode/f"{leg}-{i:02d}" for i in range(count+1)])
        corr = standard_state(params)
        legs["standard_state_kcal"] = corr
        for est in ("bar", "fep"):
            legs[f"binding_{est}_kcal"] = -legs["bound"][est] + legs["solvent"][est] + corr
        results[mode] = legs
    return results
